#!/usr/bin/env python3
"""
Experiment #1107: 6h Primary + 1d HTF — Fisher Transform + KAMA Trend + Volume Confirm

Hypothesis: Ehlers Fisher Transform catches reversals better than RSI in bear/range markets
(2025 test period). KAMA adapts to volatility changes better than HMA/EMA. Combined with
1d trend bias and volume confirmation, this should generate consistent trades with good
risk-adjusted returns on 6h timeframe.

Key innovations:
1. Ehlers Fisher Transform (period=9): Long when Fisher crosses above -1.5, Short when crosses below +1.5
   - Normalizes price to Gaussian distribution, catches reversals early
2. KAMA (Kaufman Adaptive MA, ER=10): Trend direction that adapts to volatility
   - Fast in trends, slow in chop - perfect for 6h multi-day swings
3. 1d HMA(21) bias: Only long if price > 1d_HMA, only short if price < 1d_HMA
4. Volume confirmation: Current volume > 0.8 * 20-period avg (filters false breakouts)
5. ATR(14) 2.5x stoploss with trailing
6. Discrete sizing: 0.0, ±0.25, ±0.30

Why this should work on 6h:
- Fisher Transform excels in range/bear markets (2022-2023, 2025 test)
- KAMA reduces whipsaws vs HMA/EMA during volatile periods
- 6h captures 2-4 day swings (30-60 trades/year target)
- Volume filter avoids fake breakouts common on lower TFs
- Looser entries than failed 6h experiments (Fisher levels -1.5/+1.5 not extreme)

Entry conditions (LOOSE to guarantee trades):
- LONG: Fisher crosses above -1.5 + price > KAMA + price > 1d_HMA + volume confirm
- SHORT: Fisher crosses below +1.5 + price < KAMA + price < 1d_HMA + volume confirm

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_kama_volume_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs noise)
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    n = len(close)
    if n < period + slow:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if volatility > 1e-10:
            er[i] = price_change / volatility
    
    # Smoothing constant
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution for clearer reversal signals
    Fisher = 0.5 * ln((1 + EHS) / (1 - EHS))
    EHS = 0.33 * 2 * ((close - LL) / (HH - LL) - 0.5) + 0.67 * prev_EHS
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    trigger = np.full(n, np.nan, dtype=np.float64)
    ehs = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        highest = np.max(close[i-period+1:i+1])
        lowest = np.min(close[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range > 1e-10:
            raw_ehs = 0.67 * ((close[i] - lowest) / price_range - 0.5) + 0.33 * ehs[i-1]
            ehs[i] = np.clip(raw_ehs, -0.99, 0.99)
            
            fisher[i] = 0.5 * np.log((1 + ehs[i]) / (1 - ehs[i]))
            trigger[i] = fisher[i-1] if i > period else fisher[i]
    
    return fisher, trigger

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_volume_ratio(volume, period=20):
    """Current volume vs rolling average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        avg_vol = np.mean(volume[i-period+1:i+1])
        if avg_vol > 1e-10:
            vol_ratio[i] = volume[i] / avg_vol
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_10 = calculate_kama(close, period=10, fast=2, slow=30)
    fisher, trigger = calculate_fisher(close, period=9)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track Fisher crosses
    prev_fisher = np.nan
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_10[i]) or np.isnan(fisher[i]) or np.isnan(trigger[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === TREND DIRECTION (KAMA) ===
        kama_bull = close[i] > kama_10[i]
        kama_bear = close[i] < kama_10[i]
        
        # === VOLUME CONFIRMATION ===
        volume_ok = vol_ratio[i] > 0.8  # At least 80% of avg volume
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = False
        fisher_cross_short = False
        
        if not np.isnan(prev_fisher):
            # Long: Fisher crosses above -1.5 from below
            if prev_fisher < -1.5 and fisher[i] >= -1.5:
                fisher_cross_long = True
            # Short: Fisher crosses below +1.5 from above
            if prev_fisher > 1.5 and fisher[i] <= 1.5:
                fisher_cross_short = True
        
        prev_fisher = fisher[i]
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG entry: Fisher cross + KAMA bull + HTF bull + volume
        if fisher_cross_long and kama_bull and htf_bull and volume_ok:
            desired_signal = SIZE_STRONG
        elif fisher_cross_long and kama_bull and htf_bull:
            desired_signal = SIZE_BASE
        elif fisher_cross_long and htf_bull and volume_ok:
            desired_signal = SIZE_BASE
        
        # SHORT entry: Fisher cross + KAMA bear + HTF bear + volume
        if fisher_cross_short and kama_bear and htf_bear and volume_ok:
            desired_signal = -SIZE_STRONG
        elif fisher_cross_short and kama_bear and htf_bear:
            desired_signal = -SIZE_BASE
        elif fisher_cross_short and htf_bear and volume_ok:
            desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals