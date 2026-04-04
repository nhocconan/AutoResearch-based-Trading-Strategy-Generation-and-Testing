#!/usr/bin/env python3
"""
Experiment #3787: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture swing moves aligned with weekly Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout). Volume confirmation (>1.5x) filters false breakouts. Works in bull/bear markets by adapting to weekly pivot structure. Discrete position sizing (0.25) minimizes fee drag. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3787_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly Camarilla pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly Camarilla pivot levels from prior week's OHLC
    # We need weekly OHLC - resample 1d to weekly manually but correctly
    weekly_high = np.full(len(high_1d), np.nan)
    weekly_low = np.full(len(low_1d), np.nan)
    weekly_close = np.full(len(close_1d), np.nan)
    
    # Group 1d bars into weeks (starting Monday)
    for i in range(len(close_1d)):
        if i < 4:  # Need at least 5 days for a week
            continue
        # Get past 5 trading days (approximate week)
        start_idx = max(0, i - 4)
        week_high = np.max(high_1d[start_idx:i+1])
        week_low = np.min(low_1d[start_idx:i+1])
        week_close = close_1d[i]
        
        weekly_high[i] = week_high
        weekly_low[i] = week_low
        weekly_close[i] = week_close
    
    # Calculate Camarilla levels for each week
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or np.isnan(weekly_close[i]):
            continue
        # Camarilla formula based on weekly range
        range_val = weekly_high[i] - weekly_low[i]
        camarilla_r3[i] = weekly_close[i] + range_val * 1.1 / 4
        camarilla_s3[i] = weekly_close[i] - range_val * 1.1 / 4
        camarilla_r4[i] = weekly_close[i] + range_val * 1.1 / 2
        camarilla_s4[i] = weekly_close[i] - range_val * 1.1 / 2
    
    # Align weekly Camarilla levels to 6h timeframe (shifted by 1 for completed week)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.ones(len(vol_1d))
    vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_1d[20:]
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                atr_proxy = (high[i] - low[i])
                if price < highest_since_entry - 2.5 * atr_proxy:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Donchian lower band (trend reversal)
                elif price < lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                atr_proxy = (high[i] - low[i])
                if price > lowest_since_entry + 2.5 * atr_proxy:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Donchian upper band (trend reversal)
                elif price > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average on both 6h and 1d)
        volume_spike_6h = vol_ratio[i] > 1.5
        volume_spike_1d = vol_ratio_1d_aligned[i] > 1.5
        volume_spike = volume_spike_6h and volume_spike_1d
        
        if volume_spike:
            # Determine weekly pivot context
            # Price above R3 but below R4 = bullish bias for continuation
            # Price below S3 but above S4 = bearish bias for continuation
            # Price between S3 and R3 = mean reversion zone
            
            # Long entry conditions:
            # 1. Breakout above Donchian upper band
            # 2. Price > weekly S3 (bullish bias) OR mean reversion from oversold
            breakout_up = price > highest_high[i-1]
            bullish_bias = price > camarilla_s3_aligned[i]
            mean_reversion_long = price < camarilla_s3_aligned[i] and price > camarilla_s4_aligned[i]
            
            # Short entry conditions:
            # 1. Breakout below Donchian lower band
            # 2. Price < weekly R3 (bearish bias) OR mean reversion from overbought
            breakout_down = price < lowest_low[i-1]
            bearish_bias = price < camarilla_r3_aligned[i]
            mean_reversion_short = price > camarilla_r3_aligned[i] and price < camarilla_r4_aligned[i]
            
            if (breakout_up and (bullish_bias or mean_reversion_long)):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif (breakout_down and (bearish_bias or mean_reversion_short)):
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals