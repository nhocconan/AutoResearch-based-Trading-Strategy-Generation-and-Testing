#!/usr/bin/env python3
"""
Experiment #3159: 6h Camarilla Pivot + 12h Volume Spike + Chop Regime Filter
HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
capture institutional reaction points. Combined with 12h volume spike (>2.0x average) 
to confirm participation and chop regime filter (CHOP > 61.8 = range, < 38.2 = trend) 
to adapt strategy: mean reversion in chop, breakout in trend. Designed for low trade 
frequency (target: 50-150 total trades over 4 years) with discrete position sizing 
(0.25) to minimize fee drag. Works in both bull/bear via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3159_6h_camarilla12h_vol_chop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume spike and chop regime (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h ATR(14) for chop calculation
    tr1_12h = high_12h[1:] - low_12h[1:]
    tr2_12h = np.abs(high_12h[1:] - close_12h[:-1])
    tr3_12h = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # 12h Choppy Index (CHOP) - 14 period
    def chop(high_arr, low_arr, close_arr, atr_arr, period=14):
        sum_tr = pd.Series(atr_arr).rolling(window=period, min_periods=period).sum().values
        highest_high = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        chop_val = 100 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(period)
        return chop_val
    
    chop_12h = chop(high_12h, low_12h, close_12h, atr_12h, 14)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # 12h Volume MA(20) for spike detection
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = np.ones(len(volume_12h))
    vol_ratio_12h[20:] = volume_12h[20:] / vol_ma_12h[20:]
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # === 6h Indicators: Camarilla Pivot Levels from daily data ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    daily_range = high_1d - low_1d
    camarilla_r4 = close_1d + 1.5 * daily_range
    camarilla_r3 = close_1d + 1.1 * daily_range
    camarilla_s3 = close_1d - 1.1 * daily_range
    camarilla_s4 = close_1d - 1.5 * daily_range
    
    # Align Camarilla levels to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(chop_12h_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches opposite Camarilla level (take profit)
                elif price >= s3_6h[i]:  # Long TP at S3
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches opposite Camarilla level (take profit)
                elif price <= r3_6h[i]:  # Short TP at R3
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) for confirmation
        volume_spike = vol_ratio_12h_aligned[i] > 2.0
        
        if volume_spike:
            # Regime filter: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend (breakout)
            is_range = chop_12h_aligned[i] > 61.8
            is_trend = chop_12h_aligned[i] < 38.2
            
            if is_range:
                # In choppy market: mean reversion at extreme Camarilla levels (R3/S3)
                # Long when price rejects S3 with volume spike
                if price <= s3_6h[i] and low[i] < s3_6h[i]:  # touched or broke S3
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = SIZE
                # Short when price rejects R3 with volume spike
                elif price >= r3_6h[i] and high[i] > r3_6h[i]:  # touched or broke R3
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            elif is_trend:
                # In trending market: breakout continuation at extreme Camarilla levels (R4/S4)
                # Long when price breaks R4 with volume spike (bullish breakout)
                if price > r4_6h[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = SIZE
                # Short when price breaks S4 with volume spike (bearish breakout)
                elif price < s4_6h[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:
                # Neutral chop (38.2 <= CHOP <= 61.8) - no trade
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals