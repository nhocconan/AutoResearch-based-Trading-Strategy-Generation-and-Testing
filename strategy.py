#!/usr/bin/env python3
"""
Experiment #1327: 6h Camarilla Pivot + 1d/1w Trend + Volume Confirmation
HYPOTHESIS: Camarilla pivot levels on 6h timeframe provide high-probability reversal/breakout zones. 
Trend filter from 1d timeframe ensures alignment with intermediate-term momentum. 
Weekly trend filter adds higher-timeframe bias. Volume confirmation (>1.3x average) filters for participation. 
Long at S1/S2 in 1d uptrend + 1w uptrend; Short at R1/R2 in 1d downtrend + 1w downtrend. 
Breakout continuation at R3/S3 with volume spike. Designed to work in both bull (buy dips) and bear (sell rallies) markets. 
Uses ATR-based stoploss for risk management. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1327_6h_camarilla_1d_1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # 1d trend: price > previous close = uptrend, < = downtrend
    trend_1d = np.zeros(len(close_1d))
    trend_1d[1:] = np.where(close_1d[1:] > close_1d[:-1], 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === HTF: 1w data for higher trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # 1w trend: price > previous close = uptrend, < = downtrend
    trend_1w = np.zeros(len(close_1w))
    trend_1w[1:] = np.where(close_1w[1:] > close_1w[:-1], 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 6h Indicators: Camarilla Pivot Levels (based on previous bar) ===
    # Camarilla uses previous period's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # avoid NaN on first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 4.0)
    s3 = pivot - (range_hl * 1.1 / 4.0)
    r4 = pivot + (range_hl * 1.1 / 2.0)
    s4 = pivot - (range_hl * 1.1 / 2.0)
    # Inner levels for reversal
    r1 = pivot + (range_hl * 1.1 / 12.0)
    s1 = pivot - (range_hl * 1.1 / 12.0)
    r2 = pivot + (range_hl * 1.1 / 6.0)
    s2 = pivot - (range_hl * 1.1 / 6.0)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20  # sufficient for volume MA and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(trend_1w_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(r3[i]) or np.isnan(s3[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.3x average)
        volume_spike = vol_ratio[i] > 1.3
        
        if volume_spike:
            # 1d and 1w trend alignment
            trend_1d_val = trend_1d_aligned[i]
            trend_1w_val = trend_1w_aligned[i]
            
            # Long conditions: 1d uptrend + 1w uptrend
            if trend_1d_val > 0 and trend_1w_val > 0:
                # Reversal long at S1/S2 (mean reversion in trend)
                if price <= s2[i] and price > s3[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Breakout long at R3 with volume (continuation)
                elif price >= r3[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
            
            # Short conditions: 1d downtrend + 1w downtrend
            elif trend_1d_val < 0 and trend_1w_val < 0:
                # Reversal short at R1/R2 (mean reversion in trend)
                if price >= r2[i] and price < r3[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                # Breakout short at S3 with volume (continuation)
                elif price <= s3[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
        
        # Default: no signal
        if not in_position:
            signals[i] = 0.0
    
    return signals