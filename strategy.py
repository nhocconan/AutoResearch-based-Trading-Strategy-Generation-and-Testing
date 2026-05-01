#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and ADX trend filter
# Uses 1d Camarilla pivot levels (R3/S3) as significant support/resistance
# Breakouts above R3 or below S3 with volume confirmation (>1.5x 20 EMA) and ADX>25
# Designed for low trade frequency: ~15-25 trades/year per symbol with 0.25 sizing
# Camarilla R3/S3 represent strong intraday support/resistance that often holds
# Volume spike confirms institutional participation, ADX ensures trending environment
# Works in both bull and bear markets by trading breakouts in direction of momentum

name = "12h_Camarilla_R3S3_1dVolume_ADX_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivot and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot points (R3, S3 levels)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot point
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Daily range
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3 = pivot + (range * 1.1/4), S3 = pivot - (range * 1.1/4)
    camarilla_r3 = pivot_1d + (range_1d * 1.1 / 4)
    camarilla_s3 = pivot_1d - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe (no extra delay needed for pivot points)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d ADX(14) for trend filter
    # Calculate True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = -pd.Series(low_1d).diff()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    atr_14 = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_plus_14 = dm_plus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_minus_14 = dm_minus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_14 / atr_14)
    di_minus = 100 * (dm_minus_14 / atr_14)
    
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # 1d volume confirmation: volume > 1.5 * 20-period EMA
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (1.5 * vol_ema_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient 1d data for ADX (14+14+14=42 days min) + Camarilla
    start_idx = max(42, 20)  # 42 days for ADX, 20 for volume EMA
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: break above R3 with volume spike and ADX>25
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_spike_aligned[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume spike and ADX>25
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_spike_aligned[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price re-enters Camarilla H3-L3 range (mean reversion signal)
            # Calculate H3 and L3 for exit condition
            range_1d_today = high_1d[i] - low_1d[i] if i < len(high_1d) else range_1d[-1]
            pivot_1d_today = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0 if i < len(high_1d) else pivot_1d[-1]
            camarilla_h3 = pivot_1d_today + (range_1d_today * 1.1 / 4)
            camarilla_l3 = pivot_1d_today - (range_1d_today * 1.1 / 4)
            
            # Align today's H3/L3 (use current day's values)
            if i < len(high_1d):
                camarilla_h3_today = camarilla_h3
                camarilla_l3_today = camarilla_l3
            else:
                camarilla_h3_today = camarilla_h3[-1] if len(camarilla_h3) > 0 else 0
                camarilla_l3_today = camarilla_l3[-1] if len(camarilla_l3) > 0 else 0
            
            # For simplicity, exit when price returns to pivot level (mean reversion)
            if close[i] <= pivot_1d_today if i < len(high_1d) else close[i] <= pivot_1d[-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price re-enters Camarilla H3-L3 range (mean reversion signal)
            range_1d_today = high_1d[i] - low_1d[i] if i < len(high_1d) else range_1d[-1]
            pivot_1d_today = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0 if i < len(high_1d) else pivot_1d[-1]
            camarilla_h3 = pivot_1d_today + (range_1d_today * 1.1 / 4)
            camarilla_l3 = pivot_1d_today - (range_1d_today * 1.1 / 4)
            
            if i < len(high_1d):
                camarilla_h3_today = camarilla_h3
                camarilla_l3_today = camarilla_l3
            else:
                camarilla_h3_today = camarilla_h3[-1] if len(camarilla_h3) > 0 else 0
                camarilla_l3_today = camarilla_l3[-1] if len(camarilla_l3) > 0 else 0
            
            # For simplicity, exit when price returns to pivot level (mean reversion)
            if close[i] >= pivot_1d_today if i < len(high_1d) else close[i] >= pivot_1d[-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals