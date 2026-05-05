#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot Breakout with 1d Volume Spike and Weekly Trend Filter
# Long when: Price breaks above R4 AND 1d volume > 1.5 * 20-period average volume AND weekly close > weekly open (bullish week)
# Short when: Price breaks below S4 AND 1d volume > 1.5 * 20-period average volume AND weekly close < weekly open (bearish week)
# Exit when price returns to the 1d VWAP (mean reversion to institutional value area)
# Uses Camarilla's extreme levels (R4/S4) for high-probability breakouts, volume spike for confirmation, weekly trend for bias
# Works in both bull and bear markets by only taking breakouts in the direction of the weekly trend
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_Camarilla_R4S4_Breakout_VolumeSpike_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 1d data ONCE before loop for volume average and VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d average volume (20-period)
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d VWAP for exit
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_1d = (typical_price_1d * volume_1d).cumsum() / volume_1d.cumsum()
    vwap_1d[volume_1d.cumsum() == 0] = np.nan  # Avoid division by zero
    
    # Calculate weekly trend: bullish if weekly close > weekly open
    weekly_bullish = close_1w > open_1w
    
    # Align 1d indicators to 6h timeframe
    avg_volume_20_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_20_1d)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bullish.astype(float))  # Convert bool to float for alignment
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(avg_volume_20_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(weekly_bullish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition: current 1d volume > 1.5 * 20-period average
        # We need to map 6h bar to the 1d volume - use the most recent completed 1d bar
        vol_spike = volume_1d[-1] > 1.5 * avg_volume_20_1d[-1] if len(volume_1d) > 0 and len(avg_volume_20_1d) > 0 else False
        # For simplicity and to avoid look-ahead, use aligned volume spike signal
        # Recalculate volume spike using aligned data
        vol_spike_aligned = volume[i] > 1.5 * avg_volume_20_aligned[i] if not np.isnan(avg_volume_20_aligned[i]) else False
        
        if position == 0:
            # Calculate Camarilla levels from previous 1d bar
            # Need to use previous 1d bar's OHLC to avoid look-ahead
            if len(df_1d) >= 2:
                prev_high = high_1d[-2]
                prev_low = low_1d[-2]
                prev_close = close_1d[-2]
                pivot = (prev_high + prev_low + prev_close) / 3
                range_ = prev_high - prev_low
                r4 = pivot + (range_ * 1.1 / 2)
                s4 = pivot - (range_ * 1.1 / 2)
                
                # Long: Break above R4 with volume spike and weekly bullish
                if close[i] > r4 and vol_spike_aligned and weekly_bullish_aligned[i] == 1.0:
                    signals[i] = 0.25
                    position = 1
                # Short: Break below S4 with volume spike and weekly bearish
                elif close[i] < s4 and vol_spike_aligned and weekly_bullish_aligned[i] == 0.0:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: return to 1d VWAP (mean reversion)
            if close[i] < vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to 1d VWAP (mean reversion)
            if close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals