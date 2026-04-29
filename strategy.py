#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above Camarilla R4 AND 1d ADX > 25 (strong trend) AND volume > 1.8x 20-bar avg
# Short when price breaks below Camarilla S4 AND 1d ADX > 25 AND volume > 1.8x 20-bar avg
# Exit when price retests Camarilla pivot (central level)
# Uses tighter R4/S4 levels for fewer, higher-quality breakouts
# ADX filter ensures we only trade in trending markets, reducing whipsaws in ranging conditions
# Volume confirmation adds conviction to breakouts
# Target: 15-30 trades/year on 4h timeframe (60-120 total over 4 years) to minimize fee drag
# Works in bull markets by capturing strong breakouts and in bear markets by shorting strong breakdowns
# with trend alignment preventing counter-trend trades and reducing false signals.

name = "4h_Camarilla_R4S4_Breakout_1dADX25_VolumeConf_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period) for trend filter
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_smooth = wilders_smooth(tr, period)
    dm_plus_smooth = wilders_smooth(dm_plus, period)
    dm_minus_smooth = wilders_smooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, (dm_plus_smooth / tr_smooth) * 100, 0)
    di_minus = np.where(tr_smooth != 0, (dm_minus_smooth / tr_smooth) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smooth(dx, period)  # ADX is smoothed DX
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Get 1d data for Camarilla levels (based on previous 1d bar)
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    camarilla_range = prev_high_1d - prev_low_1d
    camarilla_pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    camarilla_r4 = prev_close_1d + camarilla_range * 1.1 / 2.0  # R4 level
    camarilla_s4 = prev_close_1d - camarilla_range * 1.1 / 2.0  # S4 level
    
    # Align Camarilla levels to 4h timeframe (they represent levels from previous 1d bar)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # ADX and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_adx = adx_aligned[i]
        curr_pivot = camarilla_pivot_aligned[i]
        curr_r4 = camarilla_r4_aligned[i]
        curr_s4 = camarilla_s4_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests Camarilla pivot
            if curr_close <= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retests Camarilla pivot
            if curr_close >= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Only trade when ADX > 25 (strong trend)
            if curr_adx > 25:
                # Long when price breaks above Camarilla R4 AND volume confirmation
                if curr_close > curr_r4 and vol_conf:
                    signals[i] = 0.25
                    position = 1
                # Short when price breaks below Camarilla S4 AND volume confirmation
                elif curr_close < curr_s4 and vol_conf:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # No trend, stay flat
    
    return signals