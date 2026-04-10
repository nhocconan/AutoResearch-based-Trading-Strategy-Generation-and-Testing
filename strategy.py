#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and 1w ADX trend filter
# - Long when price breaks above 12h Camarilla R3 level AND 1d volume > 1.1x 20-period volume SMA AND 1w ADX > 25
# - Short when price breaks below 12h Camarilla S3 level AND 1d volume > 1.1x 20-period volume SMA AND 1w ADX > 25
# - Exit: price retreats to Camarilla pivot point (PP) or volume drops below average
# - Uses 12h timeframe as primary for lower trade frequency (target: 12-37 trades/year)
# - Camarilla levels from 1d for structure, 1d volume for confirmation, 1w ADX for regime filter
# - Position sizing: 0.25 discrete level to minimize fee drag

name = "12h_1d_1w_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d Camarilla pivot levels
    # Camarilla formula: PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1/4
    # S3 = PP - (H - L) * 1.1/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_range = high_1d - low_1d
    camarilla_r3 = camarilla_pp + camarilla_range * 1.1 / 4.0
    camarilla_s3 = camarilla_pp - camarilla_range * 1.1 / 4.0
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1w ADX for trend filter (regime: only trade when trending)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    atr_1w = WilderSmooth(tr, period)
    dm_plus_smooth = WilderSmooth(dm_plus, period)
    dm_minus_smooth = WilderSmooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr_1w != 0, dm_plus_smooth / atr_1w * 100, 0)
    di_minus = np.where(atr_1w != 0, dm_minus_smooth / atr_1w * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1w = WilderSmooth(dx, period)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate 12h volume SMA for confirmation
    volume_sma_20_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(adx_1w_aligned[i]) or
            np.isnan(volume_sma_20_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.1x 20-period volume SMA AND 1d volume > 1.1x 20-period volume SMA
        vol_confirm_12h = volume[i] > 1.1 * volume_sma_20_12h[i]
        vol_confirm_1d = volume_1d[i // 2] > 1.1 * volume_sma_20_1d_aligned[i] if i // 2 < len(volume_1d) else False
        vol_confirm = vol_confirm_12h and vol_confirm_1d
        
        # Trend filter: 1w ADX > 25 (trending market)
        trending = adx_1w_aligned[i] > 25
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_r3_aligned[i-1]  # Break above previous R3
        breakout_down = close[i] < camarilla_s3_aligned[i-1]  # Break below previous S3
        
        # Exit conditions: price retreats to pivot point or loss of volume confirmation
        exit_long = close[i] < camarilla_pp_aligned[i] or not vol_confirm
        exit_short = close[i] > camarilla_pp_aligned[i] or not vol_confirm
        
        if position == 0:  # Flat - look for entry
            if breakout_up and trending and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif breakout_down and trending and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals