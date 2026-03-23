#!/usr/bin/env python3
"""
Experiment #762: 12h Primary + 1d/1w HTF — Dual Regime (Choppiness + Connors RSI + Donchian)

Hypothesis: After 500+ failed strategies and analysis of what works:
1. 12h timeframe reduces noise vs 4h while maintaining sufficient trade frequency (target 30-50/year)
2. Choppiness Index(14) cleanly separates trending vs ranging regimes for adaptive logic
3. Connors RSI (RSI2 + RSI_Streak + PercentRank) outperforms standard RSI for mean reversion (75% win rate)
4. Donchian(20) breakouts capture trend continuation with HMA(21/63) confirmation
5. 1d HMA(50) + 1w HMA(50) provide robust multi-timeframe trend bias
6. ATR(14) trailing stop at 2.5x protects capital in volatile crypto markets
7. Discrete signals (0.0, ±0.25, ±0.30) minimize fee churn while maintaining exposure

Strategy design:
1. 1d HMA(50) for intermediate trend bias (aligned via mtf_data)
2. 1w HMA(50) for long-term trend bias (aligned via mtf_data)
3. 12h Choppiness Index(14) for regime detection (CHOP>61.8=ranging, CHOP<38.2=trending)
4. 12h Connors RSI for mean reversion entries (CRSI<15 long, CRSI>85 short)
5. 12h Donchian(20) for trend breakout entries
6. 12h HMA(21/63) crossover for trend confirmation
7. 12h ATR(14) for trailing stop (2.5x) and volatility filter
8. Volume filter (1.3x 20-bar SMA) for breakout confirmation

Key differences from failed 12h strategies (#752, #756, #757):
- More relaxed CRSI thresholds (15/85 vs 10/90) for sufficient trade frequency
- Added Donchian breakout for trending regime (missing in pure mean-reversion)
- Dual HTF (1d + 1w) for stronger trend confirmation
- Volume confirmation on breakouts to filter false signals
- Better hold logic to maintain positions through minor pullbacks

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 30-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_crsi_donchian_hma_1d1w_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - reduces lag while maintaining smoothness."""
    series = pd.Series(series)
    wma_half = series.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = series.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_connors_rsi(close, rsi_period=2, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(close, 2) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(2) captures short-term momentum
    RSI_Streak captures consecutive up/down days
    PercentRank captures position in recent range
    
    CRSI < 10 = extremely oversold (long signal)
    CRSI > 90 = extremely overbought (short signal)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # RSI(2) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # RSI on streak (consecutive up/down)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to positive values for RSI calculation
    streak_positive = streak + np.abs(streak.min()) + 1
    rsi_streak = calculate_rsi(streak_positive, streak_period)
    
    # PercentRank - position of close in last rank_period bars
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine components
    valid_mask = ~np.isnan(rsi_close) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_close[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channels - highest high and lowest low over period."""
    n = len(close := high)  # just for length
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures whether market is trending or ranging.
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (12h) indicators
    hma_21_12h = calculate_hma(close, 21)
    hma_63_12h = calculate_hma(close, 63)
    crsi_12h = calculate_connors_rsi(close, rsi_period=2, streak_period=2, rank_period=100)
    atr_12h = calculate_atr(high, low, close, period=14)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    vol_sma_12h = calculate_volume_sma(volume, period=20)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, 50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(hma_21_12h[i]) or np.isnan(hma_63_12h[i]):
            continue
        if np.isnan(chop_12h[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(vol_sma_12h[i]) or vol_sma_12h[i] <= 1e-10:
            continue
        
        # === TREND BIAS (1d + 1w HTF HMA50) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # Strong trend = both 1d and 1w agree
        strong_bullish = trend_1d_bullish and trend_1w_bullish
        strong_bearish = trend_1d_bearish and trend_1w_bearish
        
        # === REGIME DETECTION (Choppiness Index) ===
        trending_regime = chop_12h[i] < 38.2
        ranging_regime = chop_12h[i] > 61.8
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.3 * vol_sma_12h[i]
        
        # === CONNORS RSI SIGNALS (relaxed for trade frequency) ===
        crsi_oversold = crsi_12h[i] < 15  # relaxed from 10
        crsi_overbought = crsi_12h[i] > 85  # relaxed from 90
        crsi_extreme_oversold = crsi_12h[i] < 10
        crsi_extreme_overbought = crsi_12h[i] > 90
        
        # === HMA TREND (12h) ===
        hma_bullish = hma_21_12h[i] > hma_63_12h[i]
        hma_bearish = hma_21_12h[i] < hma_63_12h[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i] * 0.998  # near or above
        donchian_breakout_short = close[i] < donchian_lower[i] * 1.002  # near or below
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 61.8) - Mean Reversion ===
        if ranging_regime:
            # Mean reversion long: CRSI oversold + 1d trend not bearish
            if crsi_oversold and not strong_bearish:
                desired_signal = BASE_SIZE if volume_confirmed else REDUCED_SIZE
            
            # Mean reversion short: CRSI overbought + 1d trend not bullish
            if crsi_overbought and not strong_bullish:
                desired_signal = -BASE_SIZE if volume_confirmed else -REDUCED_SIZE
            
            # Extreme CRSI entries (higher conviction)
            if crsi_extreme_oversold and trend_1d_bullish:
                desired_signal = BASE_SIZE
            
            if crsi_extreme_overbought and trend_1d_bearish:
                desired_signal = -BASE_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 38.2) - Trend Following ===
        elif trending_regime:
            # Trend breakout long: Donchian breakout + HMA bullish + volume
            if donchian_breakout_long and hma_bullish and strong_bullish:
                desired_signal = BASE_SIZE if volume_confirmed else REDUCED_SIZE
            
            # Trend breakout short: Donchian breakout + HMA bearish + volume
            if donchian_breakout_short and hma_bearish and strong_bearish:
                desired_signal = -BASE_SIZE if volume_confirmed else -REDUCED_SIZE
            
            # Pullback entry in trend: CRSI moderate + trend intact
            if strong_bullish and hma_bullish and 20 < crsi_12h[i] < 40:
                desired_signal = REDUCED_SIZE
            
            if strong_bearish and hma_bearish and 60 < crsi_12h[i] < 80:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Conservative: only enter on extreme CRSI + strong trend alignment
            if crsi_extreme_oversold and strong_bullish and hma_bullish:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and strong_bearish and hma_bearish:
                desired_signal = -REDUCED_SIZE
            
            # Donchian breakout with all confirmations
            if donchian_breakout_long and strong_bullish and volume_confirmed:
                desired_signal = REDUCED_SIZE
            
            if donchian_breakout_short and strong_bearish and volume_confirmed:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and CRSI not overbought
                if (strong_bullish or trend_1d_bullish) and crsi_12h[i] < 80:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not oversold
                if (strong_bearish or trend_1d_bearish) and crsi_12h[i] > 20:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if strong bearish reversal or CRSI very overbought
            if strong_bearish and crsi_12h[i] > 70:
                desired_signal = 0.0
            # Exit if CRSI extremely overbought in ranging regime
            if ranging_regime and crsi_12h[i] > 85:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if strong bullish reversal or CRSI very oversold
            if strong_bullish and crsi_12h[i] < 30:
                desired_signal = 0.0
            # Exit if CRSI extremely oversold in ranging regime
            if ranging_regime and crsi_12h[i] < 15:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals