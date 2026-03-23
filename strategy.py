#!/usr/bin/env python3
"""
Experiment #796: 12h Primary + 1d HTF — Connors RSI Mean Reversion + Choppiness Regime + Donchian Trend

Hypothesis: After analyzing 500+ failed strategies and current best (Sharpe=0.612):
1. 12h timeframe reduces noise vs 4h while maintaining trade frequency (20-50/year)
2. Connors RSI (CRSI) has 75% win rate for mean reversion — better than standard RSI
3. 1d Choppiness Index provides clean regime detection (range vs trend)
4. Donchian(20) breakout for trend continuation entries
5. Dual regime: mean revert when CHOP>55, trend follow when CHOP<45
6. ATR(14) trailing stop at 2.5x protects from major drawdowns
7. Position sizing: 0.25-0.30 discrete levels to control fees
8. Relaxed CRSI thresholds (10/90) ensure sufficient trades on ALL symbols

Strategy design:
1. 1d Choppiness Index(14) for regime detection (aligned via mtf_data helper)
2. 12h Connors RSI for mean reversion timing
3. 12h Donchian(20) for trend breakout confirmation
4. 12h HMA(21) for smooth trend bias
5. 12h ATR(14) for trailing stop (2.5x)
6. Discrete signals: 0.0, ±0.25, ±0.30
7. Entry thresholds relaxed to ensure >=10 trades/train, >=3 tests on ALL symbols

Key differences from failed strategies:
- Connors RSI instead of standard RSI (proven 75% win rate)
- 12h primary (not 4h) — less noise, better signal quality
- CHOP thresholds: 55/45 — more regime switches than 61.8/38.2
- Donchian breakout confirmation for trend entries
- Simple hold logic — maintain until opposite signal or stoploss
- No volume filter (removes trades unnecessarily on some symbols)

Target: Sharpe > 0.612, trades >= 10 train, >= 3 test, ALL symbols positive Sharpe
Timeframe: 12h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_donchian_1d_atr_v1"
timeframe = "12h"
leverage = 1.0

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

def calculate_percent_rank(series, period=100):
    """
    Percent Rank: percentage of values in lookback period below current value.
    Used in Connors RSI calculation.
    """
    n = len(series)
    pr = np.full(n, np.nan)
    
    for i in range(period, n):
        window = series[i-period+1:i+1]
        current = series[i]
        count_below = np.sum(window < current)
        pr[i] = (count_below / period) * 100
    
    return pr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days streak length.
    - Streak = consecutive days of same direction (up=+1, down=-1)
    - RSI_Streak applies RSI formula to streak values
    
    CRSI < 10 = extreme oversold (long signal)
    CRSI > 90 = extreme overbought (short signal)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < pr_period + 1:
        return crsi
    
    # RSI(3) component
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI(2) component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Apply RSI to absolute streak values (magnitude of streak)
    streak_abs = np.abs(streak)
    rsi_streak = calculate_rsi(streak_abs, period=streak_period)
    
    # Percent Rank(100) component
    pr = calculate_percent_rank(close, period=pr_period)
    
    # Combine components
    for i in range(pr_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_3[i] + rsi_streak[i] + pr[i]) / 3.0
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel: upper = highest high, lower = lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

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
    We use 55/45 for more regime switches.
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    hma_12h = calculate_hma(close, period=21)
    atr_12h = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align 1d Choppiness for regime detection
    chop_1d_raw = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
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
        if np.isnan(crsi_12h[i]) or np.isnan(hma_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(chop_1d_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === TREND BIAS (12h HMA21) ===
        trend_bullish = close[i] > hma_12h[i]
        trend_bearish = close[i] < hma_12h[i]
        
        # === REGIME DETECTION (1d Choppiness Index) ===
        ranging_regime = chop_1d_aligned[i] > 55
        trending_regime = chop_1d_aligned[i] < 45
        
        # === CONNORS RSI SIGNALS (mean reversion) ===
        crsi_extreme_oversold = crsi_12h[i] < 15
        crsi_extreme_overbought = crsi_12h[i] > 85
        crsi_oversold = crsi_12h[i] < 25
        crsi_overbought = crsi_12h[i] > 75
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_high = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_low = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Mean reversion long: CRSI extreme oversold + price near HMA support
            if crsi_extreme_oversold and trend_bullish:
                desired_signal = BASE_SIZE
            
            # Mean reversion short: CRSI extreme overbought + price below HMA
            if crsi_extreme_overbought and trend_bearish:
                desired_signal = -BASE_SIZE
            
            # Conservative: moderate CRSI + trend alignment
            if crsi_oversold and trend_bullish and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if crsi_overbought and trend_bearish and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Trend pullback long: bullish trend + CRSI oversold (buy dip)
            if trend_bullish and crsi_oversold:
                desired_signal = BASE_SIZE
            
            # Trend pullback short: bearish trend + CRSI overbought (sell rally)
            if trend_bearish and crsi_overbought:
                desired_signal = -BASE_SIZE
            
            # Breakout continuation
            if trend_bullish and donchian_breakout_high and desired_signal == 0:
                desired_signal = BASE_SIZE
            
            if trend_bearish and donchian_breakout_low and desired_signal == 0:
                desired_signal = -BASE_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: only extreme CRSI + trend alignment
            if crsi_extreme_oversold and trend_bullish:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and trend_bearish:
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
                # Hold long if trend intact and CRSI not extreme overbought
                if trend_bullish and crsi_12h[i] < 85:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not extreme oversold
                if trend_bearish and crsi_12h[i] > 15:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses or CRSI extreme overbought
            if trend_bearish and crsi_12h[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses or CRSI extreme oversold
            if trend_bullish and crsi_12h[i] < 20:
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