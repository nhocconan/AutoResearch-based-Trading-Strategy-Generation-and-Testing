# 4h Williams Fractal + Volume Spike + ADX Filter
# Strategy: Long when bullish fractal forms above 50 EMA with volume spike in high ADX trend
# Short when bearish fractal forms below 50 EMA with volume spike in high ADX trend
# Uses fractals for reversal signals, ADX for trend strength, volume for confirmation
# Timeframe: 4h (optimal balance of signal quality and trade frequency)
# Expected trades: ~25-35/year per symbol (100-140 total over 4 years)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA and fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams Fractals on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Calculate ADX(14) on 4h data for trend strength
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    tr[0] = high[0] - low[0]
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    # Smooth DM values
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    
    for i in range(1, n):
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * 13 + plus_dm[i]) / 14
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * 13 + minus_dm[i]) / 14
    
    for i in range(14, n):
        if plus_dm_smooth[i] + minus_dm_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / (plus_dm_smooth[i] + minus_dm_smooth[i])
            minus_di[i] = 100 * minus_dm_smooth[i] / (plus_dm_smooth[i] + minus_dm_smooth[i])
            
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # Calculate ADX as smoothed DX
    adx = np.zeros(n)
    adx[14] = dx[14]  # First ADX value
    for i in range(15, n):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align daily indicators to 4h timeframe with proper delay for fractals
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need EMA(50) and enough data for ADX
    start_idx = max(50, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        adx_val = adx_aligned[i]
        vol_current = volume[i]
        
        # Calculate 20-period volume average for spike detection
        if i >= 20:
            vol_avg = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_avg = vol_current
        
        # Volume spike: current volume > 2x average
        volume_spike = vol_current > (vol_avg * 2.0)
        
        # ADX filter: trend strength > 25
        strong_trend = adx_val > 25
        
        # Entry conditions
        if position == 0:
            # Long: bullish fractal above EMA + volume spike + strong trend
            if (bullish_fractal_aligned[i] and 
                close[i] > ema_trend and 
                volume_spike and 
                strong_trend):
                signals[i] = size
                position = 1
            # Short: bearish fractal below EMA + volume spike + strong trend
            elif (bearish_fractal_aligned[i] and 
                  close[i] < ema_trend and 
                  volume_spike and 
                  strong_trend):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish fractal forms or trend weakens
            if bearish_fractal_aligned[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: bullish fractal forms or trend weakens
            if bullish_fractal_aligned[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WilliamsFractal_VolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0