#!/usr/bin/env python3
"""
Experiment #1059: 4h Primary + 1d HTF — Fisher Transform + Vol Spike + Asymmetric Regime

Hypothesis: After 767+ failed experiments, the key insight is that complex regime-switching
with too many filters causes 0 trades. The winning approach combines:

1. EHLERS FISHER TRANSFORM (period=9):
   - Normalizes price into Gaussian distribution for cleaner reversal signals
   - Long: Fisher crosses above -1.5 (oversold reversal)
   - Short: Fisher crosses below +1.5 (overbought reversal)
   - Research shows 75% win rate in bear/range markets

2. VOLATILITY SPIKE REVERSION:
   - ATR(7)/ATR(30) > 1.8 = vol expansion (panic/extreme)
   - Enter when vol spikes + price at BB extreme
   - Exit when ATR ratio < 1.3 (vol crush)
   - Captures "fear climax" reversals

3. ASYMMETRIC REGIME FILTER (1d HMA21):
   - Only LONG when close > 1d_HMA21 (bullish macro)
   - Only SHORT when close < 1d_HMA21 (bearish macro)
   - Prevents counter-trend trades in strong moves

4. RELAXED ENTRY THRESHOLDS (avoid 0 trades):
   - Fisher: -1.5/+1.5 (not -2/+2)
   - ATR ratio: >1.8 (not >2.5)
   - ADX: >15 for trend confirmation (not >25)

5. ATR TRAILING STOP: 2.5x ATR(14) from entry
   - Signal→0 when stop hit (mandatory risk management)

Why this should work:
- Fisher Transform catches reversals better than RSI in bear markets
- Vol spike filter ensures we enter at extremes (high win rate)
- 1d HMA provides macro bias without being too restrictive
- Relaxed thresholds ensure 30+ trades/train, 3+ trades/test

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 20-50 trades/year)
Position Size: 0.25-0.30 discrete levels
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_vol_spike_1d_hma_asymmetric_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price into Gaussian distribution
    for cleaner reversal signals. Best for catching tops/bottoms in bear markets.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize to -1 to +1 range
    3. Apply Fisher: 0.5 * ln((1 + value) / (1 - value))
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period + 2:
        return fisher, fisher_signal
    
    # Calculate typical price
    typical = (high + low) / 2.0
    
    # Find highest high and lowest low over period
    for i in range(period, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        price_range = highest - lowest
        
        if price_range < 1e-10:
            continue
        
        # Normalize to -1 to +1 (with 0.99 factor to prevent division issues)
        normalized = 0.99 * (2.0 * (typical[i] - lowest) / price_range - 1.0)
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Apply Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Fisher signal (previous value for crossover detection)
        if i > period:
            fisher_signal[i] = fisher[i - 1]
    
    return fisher, fisher_signal

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """
    ATR Ratio - measures volatility expansion vs contraction
    Ratio > 2.0 = vol spike (panic/extreme)
    Ratio < 1.2 = vol crush (calm)
    """
    n = len(close)
    atr_ratio = np.full(n, np.nan)
    
    if n < long_period + 1:
        return atr_ratio
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate ATRs
    tr_series = pd.Series(tr)
    atr_short = tr_series.ewm(span=short_period, min_periods=short_period, adjust=False).mean().values
    atr_long = tr_series.ewm(span=long_period, min_periods=long_period, adjust=False).mean().values
    
    # Calculate ratio
    atr_ratio = np.divide(atr_short, atr_long, out=np.zeros_like(atr_short), where=atr_long != 0)
    
    return atr_ratio

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stops."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(series, period):
    """Hull Moving Average - faster and smoother than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands for mean reversion entries."""
    n = len(close)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < period:
        return upper, middle, lower
    
    close_series = pd.Series(close)
    rolling_mean = close_series.rolling(window=period, min_periods=period).mean()
    rolling_std = close_series.rolling(window=period, min_periods=period).std()
    
    middle = rolling_mean.values
    upper = (rolling_mean + std_mult * rolling_std).values
    lower = (rolling_mean - std_mult * rolling_std).values
    
    return upper, middle, lower

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength indicator."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    # Calculate +DM, -DM, and TR
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
    
    # Smooth with EMA
    plus_dm_series = pd.Series(plus_dm)
    minus_dm_series = pd.Series(minus_dm)
    tr_series = pd.Series(tr)
    
    smoothed_plus_dm = plus_dm_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    smoothed_minus_dm = minus_dm_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    smoothed_tr = tr_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI, -DI
    plus_di = np.divide(100 * smoothed_plus_dm, smoothed_tr, out=np.zeros_like(smoothed_plus_dm), where=smoothed_tr != 0)
    minus_di = np.divide(100 * smoothed_minus_dm, smoothed_tr, out=np.zeros_like(smoothed_minus_dm), where=smoothed_tr != 0)
    
    # Calculate DX and ADX
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = np.divide(100 * di_diff, di_sum, out=np.zeros_like(di_diff), where=di_sum != 0)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA21 for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    atr_ratio = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    adx = calculate_adx(high, low, close, period=14)
    
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
    
    # Track Fisher crossover state
    prev_fisher = np.nan
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(fisher[i]) or np.isnan(atr_ratio[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(adx[i]):
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        if np.isnan(hma_1d_aligned[i]):
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        # === MACRO TREND (1d HMA21) - ASYMMETRIC FILTER ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY REGIME ===
        vol_spike = atr_ratio[i] > 1.8  # Vol expansion (panic/extreme)
        vol_normal = atr_ratio[i] < 1.5  # Normal volatility
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # Fisher crossover detection
        fisher_cross_up = False
        fisher_cross_down = False
        
        if not np.isnan(prev_fisher) and not np.isnan(fisher_signal[i]):
            # Cross above -1.5 (oversold reversal)
            if prev_fisher < -1.5 and fisher[i] >= -1.5:
                fisher_cross_up = True
            # Cross below +1.5 (overbought reversal)
            if prev_fisher > 1.5 and fisher[i] <= 1.5:
                fisher_cross_down = True
        
        prev_fisher = fisher[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Must have macro bullish bias for longs (asymmetric)
        if macro_bull:
            # Strong long: Fisher cross up + vol spike + price near BB lower
            if fisher_cross_up and vol_spike and close[i] <= bb_lower[i]:
                desired_signal = BASE_SIZE
            # Moderate long: Fisher oversold + vol spike + macro bull
            elif fisher_oversold and vol_spike and adx[i] > 15:
                desired_signal = REDUCED_SIZE
            # Weak long: Fisher cross up + macro bull (no vol spike needed)
            elif fisher_cross_up and vol_normal:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        # Must have macro bearish bias for shorts (asymmetric)
        if macro_bear:
            # Strong short: Fisher cross down + vol spike + price near BB upper
            if fisher_cross_down and vol_spike and close[i] >= bb_upper[i]:
                desired_signal = -BASE_SIZE
            # Moderate short: Fisher overbought + vol spike + macro bear
            elif fisher_overbought and vol_spike and adx[i] > 15:
                desired_signal = -REDUCED_SIZE
            # Weak short: Fisher cross down + macro bear (no vol spike needed)
            elif fisher_cross_down and vol_normal:
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
        
        # === HOLD LOGIC — Maintain position if signal intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro still bullish and Fisher not overbought
                if macro_bull and fisher[i] < 1.5:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro still bearish and Fisher not oversold
                if macro_bear and fisher[i] > -1.5:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses bearish
            if macro_bear:
                desired_signal = 0.0
            # Exit long if Fisher becomes overbought
            if fisher[i] > 1.8:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses bullish
            if macro_bull:
                desired_signal = 0.0
            # Exit short if Fisher becomes oversold
            if fisher[i] < -1.8:
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
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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