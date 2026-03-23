#!/usr/bin/env python3
"""
Experiment #792: 12h Primary + 1d/1w HTF — Dual Regime with Choppiness + CRSI + HMA

Hypothesis: After analyzing 500+ failed strategies and current best (Sharpe=0.612):
1. 12h timeframe balances trade frequency (20-50/year) with signal quality
2. Choppiness Index (CHOP) regime detection switches between mean reversion and trend
3. Connors RSI (CRSI) with relaxed thresholds (30/70) generates enough trades
4. 1w HMA(21) + 1d HMA(21) provide strong trend bias without being too slow
5. 12h HMA(16/48) crossover for trend confirmation on primary TF
6. ATR(14) trailing stop at 2.5x protects from major drawdowns
7. Position sizing: 0.25-0.30 discrete levels to control fees

Strategy design:
1. 1w HMA(21) + 1d HMA(21) for long-term trend bias (aligned via mtf_data helper)
2. 12h Choppiness Index(14) for regime detection
3. 12h Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
4. 12h HMA(16) vs HMA(48) crossover for trend confirmation
5. 12h ATR(14) for trailing stop (2.5x)
6. 12h Bollinger Bands(20, 2.0) for mean reversion bounds
7. Relaxed entry thresholds to ensure >=10 trades/train, >=3 trades/test

Key differences from failed 12h strategies:
- CRSI thresholds: 30/70 (not 10/90 or 25/75) — generates more trades
- CHOP thresholds: 55/45 (not 61.8/38.2) — more regime switches
- Dual HMA crossover (16/48) for trend confirmation
- Volume filter: 1.2x (not 1.5x) — less restrictive
- Hold logic: maintain position until opposite signal or stoploss

Target: Sharpe > 0.612, trades >= 10 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_crsi_hma_1d1w_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_ema(series, period):
    """Exponential Moving Average."""
    return pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
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

def calculate_rsi_streak(close, period=2):
    """Connors RSI Streak component."""
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 5:
        return streak_rsi
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_abs = np.abs(streak)
    streak_score = np.where(streak >= 0, streak_abs, -streak_abs)
    streak_rsi = calculate_rsi(streak_score + 100, period)
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Connors RSI Percent Rank component."""
    n = len(close)
    pct_rank = np.full(n, np.nan)
    
    if n < period + 1:
        return pct_rank
    
    returns = np.diff(close) / (close[:-1] + 1e-10) * 100
    returns = np.concatenate([[0], returns])
    
    for i in range(period, n):
        window = returns[i-period:i]
        current = returns[i]
        pct_rank[i] = 100 * np.sum(window < current) / period
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pct_rank_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pct_rank_period)
    
    with np.errstate(invalid='ignore'):
        crsi = (rsi_short + streak_rsi + pct_rank) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands."""
    sma = calculate_sma(close, period)
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending.
    We use 55/45 for more regime switches on 12h.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr_sum += max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
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
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, pct_rank_period=100)
    atr_12h = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger(close, period=20, std_mult=2.0)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    vol_sma_12h = calculate_volume_sma(volume, period=20)
    
    # Calculate HMA crossover on primary TF
    hma_16_12h = calculate_hma(close, 16)
    hma_48_12h = calculate_hma(close, 48)
    
    # Calculate and align HTF HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(bb_sma[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(chop_12h[i]):
            continue
        if np.isnan(vol_sma_12h[i]) or vol_sma_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_16_12h[i]) or np.isnan(hma_48_12h[i]):
            continue
        
        # === TREND BIAS (1w and 1d HTF HMA21) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # Strong trend when both HTF agree
        strong_bullish = trend_1w_bullish and trend_1d_bullish
        strong_bearish = trend_1w_bearish and trend_1d_bearish
        
        # === REGIME DETECTION (Choppiness Index) ===
        ranging_regime = chop_12h[i] > 55
        trending_regime = chop_12h[i] < 45
        neutral_regime = not ranging_regime and not trending_regime
        
        # === VOLUME CONFIRMATION (relaxed) ===
        volume_confirmed = volume[i] > 1.2 * vol_sma_12h[i]
        
        # === CRSI SIGNALS (relaxed thresholds for more trades) ===
        crsi_oversold = crsi_12h[i] < 30
        crsi_overbought = crsi_12h[i] > 70
        crsi_extreme_oversold = crsi_12h[i] < 20
        crsi_extreme_overbought = crsi_12h[i] > 80
        crsi_neutral_low = 35 < crsi_12h[i] < 50
        crsi_neutral_high = 50 < crsi_12h[i] < 65
        
        # === HMA CROSSOVER ===
        hma_bullish_cross = hma_16_12h[i] > hma_48_12h[i]
        hma_bearish_cross = hma_16_12h[i] < hma_48_12h[i]
        
        # === BOLLINGER POSITION ===
        below_bb_lower = close[i] < bb_lower[i]
        above_bb_upper = close[i] > bb_upper[i]
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Mean reversion long: CRSI oversold + below BB lower
            if crsi_oversold and below_bb_lower:
                desired_signal = BASE_SIZE if volume_confirmed else REDUCED_SIZE
            
            # Mean reversion short: CRSI overbought + above BB upper
            elif crsi_overbought and above_bb_upper:
                desired_signal = -BASE_SIZE if volume_confirmed else -REDUCED_SIZE
            
            # Conservative: extreme CRSI + HTF trend alignment
            elif crsi_extreme_oversold and (trend_1w_bullish or trend_1d_bullish):
                desired_signal = REDUCED_SIZE
            
            elif crsi_extreme_overbought and (trend_1w_bearish or trend_1d_bearish):
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Trend pullback long: HTF bullish + CRSI neutral low + HMA bullish
            if strong_bullish and crsi_neutral_low and hma_bullish_cross:
                desired_signal = BASE_SIZE if volume_confirmed else REDUCED_SIZE
            
            # Trend pullback short: HTF bearish + CRSI neutral high + HMA bearish
            elif strong_bearish and crsi_neutral_high and hma_bearish_cross:
                desired_signal = -BASE_SIZE if volume_confirmed else -REDUCED_SIZE
            
            # Breakout continuation with volume
            elif strong_bullish and above_bb_upper and volume_confirmed:
                desired_signal = BASE_SIZE
            
            elif strong_bearish and below_bb_lower and volume_confirmed:
                desired_signal = -BASE_SIZE
            
            # HMA crossover entry
            elif hma_bullish_cross and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
            
            elif hma_bearish_cross and trend_1d_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: only extreme CRSI + HTF trend alignment
            if crsi_extreme_oversold and (trend_1w_bullish or trend_1d_bullish):
                desired_signal = REDUCED_SIZE
            
            elif crsi_extreme_overbought and (trend_1w_bearish or trend_1d_bearish):
                desired_signal = -REDUCED_SIZE
            
            # Also allow basic mean reversion
            elif crsi_oversold and below_bb_lower:
                desired_signal = REDUCED_SIZE
            
            elif crsi_overbought and above_bb_upper:
                desired_signal = -REDUCED_SIZE
            
            # HMA crossover as fallback
            elif hma_bullish_cross and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
            
            elif hma_bearish_cross and trend_1d_bearish:
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and CRSI not overbought
                if (trend_1w_bullish or trend_1d_bullish) and crsi_12h[i] < 75:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not oversold
                if (trend_1w_bearish or trend_1d_bearish) and crsi_12h[i] > 25:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if strong bearish reversal
            if strong_bearish and crsi_12h[i] > 70:
                desired_signal = 0.0
            # Exit if price hits BB upper in ranging regime
            if ranging_regime and above_bb_upper:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if strong bullish reversal
            if strong_bullish and crsi_12h[i] < 30:
                desired_signal = 0.0
            # Exit if price hits BB lower in ranging regime
            if ranging_regime and below_bb_lower:
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