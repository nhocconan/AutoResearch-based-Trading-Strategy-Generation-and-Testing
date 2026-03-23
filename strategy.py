#!/usr/bin/env python3
"""
Experiment #1062: 12h Primary + 1d HTF — Simplified Dual Regime with Relaxed Entries

Hypothesis: After 770+ failed experiments, the pattern is clear:
- Complex strategies (CRSI + multiple filters) generate 0 trades or negative Sharpe
- 12h timeframe naturally limits trades to 20-50/year (perfect for fee management)
- RELAXED thresholds are critical: previous 12h strategies failed due to too few trades

Strategy Design:
1. REGIME: Choppiness Index (14) - binary 55 threshold
2. TREND MODE (CHOP < 55): HMA(16/48) crossover + 1d HMA21 filter + ADX > 15
3. RANGE MODE (CHOP >= 55): RSI(14) extremes (30/70) + Bollinger bands
4. MACRO FILTER: 1d HMA21 for directional bias (asymmetric)
5. STOPLOSS: 2.5x ATR trailing (signal → 0)
6. POSITION SIZE: 0.25-0.30 discrete levels

Key Changes from Failed Experiments (#1052, #1056):
- RELAXED thresholds: ADX > 15 (not 20+), RSI 30/70 (not 25/75), CHOP 55 (not complex zones)
- SIMPLER exits: primarily stoploss or clear signal reversal (not multi-condition)
- HOLD LOGIC: maintain position if core thesis intact (reduces churn)
- 1d HMA21: smoother macro filter than 12h HMA

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 20-50 trades/year)
Position Size: 0.25-0.30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_simplified_dual_regime_1d_hma_rsi_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_choppiness_index(high, low, close, period=14):
    """Choppiness Index - market ranging vs trending."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    
    avg_gain = gain_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
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
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands."""
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
    """Average Directional Index."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
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
    
    plus_dm_series = pd.Series(plus_dm)
    minus_dm_series = pd.Series(minus_dm)
    tr_series = pd.Series(tr)
    
    smoothed_plus_dm = plus_dm_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    smoothed_minus_dm = minus_dm_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    smoothed_tr = tr_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.divide(100 * smoothed_plus_dm, smoothed_tr, out=np.zeros_like(smoothed_plus_dm), where=smoothed_tr != 0)
    minus_di = np.divide(100 * smoothed_minus_dm, smoothed_tr, out=np.zeros_like(smoothed_minus_dm), where=smoothed_tr != 0)
    
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
    
    # Calculate primary (12h) indicators
    chop = calculate_choppiness_index(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    adx = calculate_adx(high, low, close, period=14)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(chop[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(adx[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            continue
        
        # === REGIME DETECTION ===
        is_range = chop[i] >= 55.0
        is_trend = chop[i] < 55.0
        
        # === MACRO TREND (1d HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        desired_signal = 0.0
        
        # === RANGE MODE: MEAN REVERSION ===
        if is_range:
            # Long: RSI oversold + price at BB lower + macro bullish bias
            if rsi[i] < 35 and close[i] <= bb_lower[i] and macro_bull:
                desired_signal = BASE_SIZE
            # Short: RSI overbought + price at BB upper + macro bearish bias
            elif rsi[i] > 65 and close[i] >= bb_upper[i] and macro_bear:
                desired_signal = -BASE_SIZE
            # Weaker signals (relaxed for more trades)
            elif rsi[i] < 30 and macro_bull:
                desired_signal = REDUCED_SIZE
            elif rsi[i] > 70 and macro_bear:
                desired_signal = -REDUCED_SIZE
        
        # === TREND MODE: TREND FOLLOWING ===
        elif is_trend:
            # Long: HMA16 > HMA48 + price > 1d HMA + ADX shows trend
            if hma_16[i] > hma_48[i] and macro_bull and adx[i] > 15:
                desired_signal = BASE_SIZE
            # Short: HMA16 < HMA48 + price < 1d HMA + ADX shows trend
            elif hma_16[i] < hma_48[i] and macro_bear and adx[i] > 15:
                desired_signal = -BASE_SIZE
            # Weaker trend signals (relaxed for more trades)
            elif hma_16[i] > hma_48[i] and macro_bull:
                desired_signal = REDUCED_SIZE
            elif hma_16[i] < hma_48[i] and macro_bear:
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
                # Hold long if macro still bullish or range mode with RSI not overbought
                if macro_bull or (is_range and rsi[i] < 60):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro still bearish or range mode with RSI not oversold
                if macro_bear or (is_range and rsi[i] > 40):
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS (simplified for more trades) ===
        if in_position and position_side > 0:
            # Exit long if macro reverses bearish AND RSI overbought
            if macro_bear and rsi[i] > 65:
                desired_signal = 0.0
            # Exit long if trend mode and HMA crossover reverses
            if is_trend and hma_16[i] < hma_48[i]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses bullish AND RSI oversold
            if macro_bull and rsi[i] < 35:
                desired_signal = 0.0
            # Exit short if trend mode and HMA crossover reverses
            if is_trend and hma_16[i] > hma_48[i]:
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