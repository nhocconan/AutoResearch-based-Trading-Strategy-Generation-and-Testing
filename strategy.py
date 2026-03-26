#!/usr/bin/env python3
"""
Experiment #021: 4h Camarilla Pivot Break + Volume + Choppiness (Simplified)

HYPOTHESIS: Camarilla pivot levels (S3/R3) act as institutional support/resistance.
Price crossing these levels signals institutions accepting liquidity.
Volume spike confirms institutional involvement.
Choppiness < 55 filters to trending periods only.
This pattern has DB precedent: gen_camarilla_pivot_volume_spike_choppiness_4h_v1 (Sharpe=1.471, 95 trades).

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Camarilla derived from previous range, works in any market regime
- Bear: short R3 break with tight ATR stop
- Bull: long S3 break with trailing stop
- Range: mean-revert between pivots

TARGET: 75-100 total trades over 4 years (2-3/month). DB winner had 95 trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_vol_chop_simple_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        wsum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1]
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / wsum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    return pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - CHOP < 55 = trending (trade), > 55 = choppy (no trade)"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        price_range = highest - lowest
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_camarilla_pivots(prev_high, prev_low, prev_close):
    """
    Camarilla pivot levels
    S3 = close - range * 1.1/4, R3 = close + range * 1.1/4
    """
    n = len(prev_high)
    pivots = {
        's3': np.full(n, np.nan, dtype=np.float64),
        'r3': np.full(n, np.nan, dtype=np.float64),
    }
    
    for i in range(n):
        if np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]):
            continue
        
        rng = prev_high[i] - prev_low[i]
        if rng <= 1e-10:
            continue
        
        c = prev_close[i]
        pivots['s3'][i] = c - rng * 1.1 / 4
        pivots['r3'][i] = c + rng * 1.1 / 4
    
    return pivots

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data ONCE for pivots and trend
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Camarilla pivots from 1d
    cam_pivots = calculate_camarilla_pivots(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align pivots to 4h (shifted by 1 for no look-ahead)
    s3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['s3'])
    r3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['r3'])
    
    # 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume: 20-bar MA ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest = 0.0
    lowest = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        s3 = s3_aligned[i]
        r3 = r3_aligned[i]
        chop = chop_14[i]
        vol_spike = vol_ratio[i] > 1.5
        
        # Trend: price vs 1d HMA
        price_above_1d_hma = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        
        desired_signal = 0.0
        
        # === REGIME CHECK: trending only ===
        if chop < 55.0:
            # === LONG ENTRY: price breaks below S3 with volume ===
            # Price crosses below S3 (close < S3, prev close >= prev S3)
            if i > 0 and close[i] < s3 and close[i-1] >= s3:
                if vol_spike and price_above_1d_hma:
                    desired_signal = SIZE
                elif price_above_1d_hma:
                    desired_signal = SIZE
            
            # === SHORT ENTRY: price breaks above R3 with volume ===
            if i > 0 and close[i] > r3 and close[i-1] <= r3:
                if vol_spike and not price_above_1d_hma:
                    desired_signal = -SIZE
                elif not price_above_1d_hma:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2x ATR) ===
        stoploss_hit = False
        if in_position and position_side > 0:
            highest = max(highest, high[i])
            trailing_stop = highest - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_hit = True
        
        if in_position and position_side < 0:
            lowest = min(lowest, low[i])
            trailing_stop = lowest + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_hit = True
        
        if stoploss_hit:
            desired_signal = 0.0
        
        # === TAKE PROFIT: price reaches opposite pivot ===
        tp_hit = False
        if in_position and position_side > 0:
            # TP at R3
            if high[i] >= r3:
                tp_hit = True
        
        if in_position and position_side < 0:
            # TP at S3
            if low[i] <= s3:
                tp_hit = True
        
        if tp_hit:
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest = high[i]
                lowest = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals