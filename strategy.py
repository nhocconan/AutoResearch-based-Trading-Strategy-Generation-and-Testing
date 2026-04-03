#!/usr/bin/env python3
"""
Experiment #208: 12h Camarilla Pivot + Volume Spike + 1w Trend Filter
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) on 12h combined with 1w trend filter and volume confirmation captures institutional order flow. Uses 1w EMA50 for robust trend filtering to avoid whipsaws in sideways markets. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_208_12h_camarilla_pivot_volume_1w_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA50 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_up_1w = close_1w > ema50_1w
    trend_down_1w = close_1w < ema50_1w
    
    # Align to 12h timeframe
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    trend_down_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_down_1w)
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr_12h = np.zeros(n)
    tr_12h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_12h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === 12h Indicators: Camarilla Pivot Levels from previous bar ===
    camarilla_r4 = np.zeros(n)
    camarilla_r3 = np.zeros(n)
    camarilla_s3 = np.zeros(n)
    camarilla_s4 = np.zeros(n)
    camarilla_pivot = np.zeros(n)
    
    for i in range(1, n):
        # Previous bar's OHLC
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Pivot point
        camarilla_pivot[i] = (prev_high + prev_low + prev_close) / 3.0
        
        # Range
        range_ = prev_high - prev_low
        
        # Camarilla levels
        camarilla_r4[i] = camarilla_pivot[i] + range_ * 1.1 / 2.0
        camarilla_r3[i] = camarilla_pivot[i] + range_ * 1.1 / 4.0
        camarilla_s3[i] = camarilla_pivot[i] - range_ * 1.1 / 4.0
        camarilla_s4[i] = camarilla_pivot[i] - range_ * 1.1 / 2.0
    
    # For first bar, use current values (will be overwritten quickly)
    camarilla_r4[0] = camarilla_r3[0] = camarilla_pivot[0] = camarilla_s3[0] = camarilla_s4[0] = close[0]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(trend_up_1w_aligned[i]) or np.isnan(trend_down_1w_aligned[i]) or
            np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if price reaches R3 (take profit) or R4 breaks with volume (continuation)
                if price >= camarilla_r3[i] and volume_spike:
                    # Take partial profit at R3, continue if breaks R4
                    if price >= camarilla_r4[i]:
                        # Continue the trend
                        signals[i] = SIZE
                    else:
                        # Take profit at R3
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if price reaches S3 (take profit) or S4 breaks with volume (continuation)
                if price <= camarilla_s3[i] and volume_spike:
                    # Take partial profit at S3, continue if breaks S4
                    if price <= camarilla_s4[i]:
                        # Continue the trend
                        signals[i] = -SIZE
                    else:
                        # Take profit at S3
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Mean reversion at R3/S3 with volume spike and 1w trend alignment
        # Long: Price rejects R3 (comes back below) in uptrend with volume
        if (price < camarilla_r3[i] and 
            close[i-1] >= camarilla_r3[i-1] and  # Was at or above R3 previous bar
            trend_up_1w_aligned[i] and 
            volume_spike):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Price rejects S3 (goes back above) in downtrend with volume
        elif (price > camarilla_s3[i] and 
              close[i-1] <= camarilla_s3[i-1] and  # Was at or below S3 previous bar
              trend_down_1w_aligned[i] and 
              volume_spike):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        # Breakout continuation at R4/S4 with volume spike and 1w trend alignment
        # Long: Price breaks above R4 with volume in uptrend
        elif (price > camarilla_r4[i] and 
              trend_up_1w_aligned[i] and 
              volume_spike):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Price breaks below S4 with volume in downtrend
        elif (price < camarilla_s4[i] and 
              trend_down_1w_aligned[i] and 
              volume_spike):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals