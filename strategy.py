#!/usr/bin/env python3
"""
Experiment #1439: 6h Camarilla Pivot Reversal + 12h Trend Filter + Volume Confirmation
HYPOTHESIS: Camarilla pivot levels (R3/S3, R4/S4) act as intraday support/resistance on 6h timeframe. 
In ranging markets (CHOP > 50), price tends to reverse at R3/S3 levels. In trending markets (CHOP < 50), 
breaks of R4/S4 indicate continuation. 12h trend filter ensures alignment with higher timeframe momentum. 
Volume confirmation (>1.5x average) filters for meaningful participation. Designed for low trade frequency 
(50-150 total over 4 years) with discrete position sizing to minimize fee impact. Works in both bull and bear 
markets by adapting to regime via choppiness index.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1439_6h_camarilla_pivot_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # Simple trend: price > previous close = uptrend, < = downtrend
    trend_12h = np.zeros(len(close_12h))
    trend_12h[1:] = np.where(close_12h[1:] > close_12h[:-1], 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # === HTF: 1d data for choppiness regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr_1d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(period)
    # Simplified: use rolling ATR sum and range
    atr_sum_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_1d = max_high_1d - min_low_1d
    chop_1d = np.zeros(len(close_1d))
    mask = range_1d > 0
    chop_1d[mask] = 100 * np.log10(atr_sum_1d[mask] / range_1d[mask]) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 6h Indicators: Camarilla Pivot Levels from previous bar ===
    # Camarilla: based on previous bar's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # avoid NaN on first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_bar = prev_high - prev_low
    
    # Resistance levels
    r3 = pivot + (range_bar * 1.1 / 4)
    r4 = pivot + (range_bar * 1.1 / 2)
    # Support levels
    s3 = pivot - (range_bar * 1.1 / 4)
    s4 = pivot - (range_bar * 1.1 / 2)
    
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
        if (np.isnan(trend_12h_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(r3[i]) or np.isnan(r4[i]) or np.isnan(s3[i]) or np.isnan(s4[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Regime filter: CHOP > 50 = ranging (mean revert), CHOP < 50 = trending (breakout)
            is_ranging = chop_1d_aligned[i] > 50
            
            if is_ranging:
                # Ranging market: fade at R3/S3 levels
                if price <= r3[i] and trend_12h_aligned[i] > 0:  # Pullback to R3 in uptrend -> long
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                elif price >= s3[i] and trend_12h_aligned[i] < 0:  # Pullback to S3 in downtrend -> short
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:
                # Trending market: breakout continuation at R4/S4 levels
                if price >= r4[i] and trend_12h_aligned[i] > 0:  # Break above R4 in uptrend -> long
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                elif price <= s4[i] and trend_12h_aligned[i] < 0:  # Break below S4 in downtrend -> short
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals