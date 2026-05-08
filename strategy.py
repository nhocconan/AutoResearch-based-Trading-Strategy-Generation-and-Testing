# 3/4
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Aroon + 1d ADX trend filter + volume confirmation
# Aroon identifies trend strength and direction (Aroon Up > Aroon Down = uptrend).
# Strong trends occur when Aroon Up > 70 and Aroon Down < 30 (or vice versa for downtrend).
# We enter when Aroon crosses into strong trend territory, confirmed by 1d ADX > 25 and volume spike.
# Exits when Aroon weakens or ADX drops below 20.
# Targets 12-30 trades per year (~48-120 total over 4 years) to minimize fee drag.

name = "6h_Aroon_1dADX_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Aroon indicator (25-period)
    def aroon_up(high, period):
        n = len(high)
        up = np.full(n, np.nan)
        for i in range(period-1, n):
            window = high[i-period+1:i+1]
            high_idx = np.argmax(window)
            up[i] = ((period - 1 - high_idx) / (period - 1)) * 100
        return up
    
    def aroon_down(low, period):
        n = len(low)
        down = np.full(n, np.nan)
        for i in range(period-1, n):
            window = low[i-period+1:i+1]
            low_idx = np.argmin(window)
            down[i] = ((period - 1 - low_idx) / (period - 1)) * 100
        return down
    
    aroon_up_val = aroon_up(high, 25)
    aroon_down_val = aroon_down(low, 25)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        n = len(values)
        smoothed = np.full(n, np.nan)
        if n < period:
            return smoothed
        # First value is simple average
        smoothed[period-1] = np.nansum(values[1:period])
        for i in range(period, n):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    period_adx = 14
    atr = wilders_smoothing(tr, period_adx)
    dm_plus_smooth = wilders_smoothing(dm_plus, period_adx)
    dm_minus_smooth = wilders_smoothing(dm_minus, period_adx)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, (dm_plus_smooth / atr) * 100, 0)
    di_minus = np.where(atr != 0, (dm_minus_smooth / atr) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, period_adx)
    
    # Align 1d ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 1.8x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_conf = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for Aroon and ADX
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(aroon_up_val[i]) or np.isnan(aroon_down_val[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        au = aroon_up_val[i]
        ad = aroon_down_val[i]
        adx_val = adx_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: Aroon Up > 70 and Aroon Down < 30, ADX > 25, volume confirmation
            if au > 70 and ad < 30 and adx_val > 25 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Aroon Down > 70 and Aroon Up < 30, ADX > 25, volume confirmation
            elif ad > 70 and au < 30 and adx_val > 25 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Aroon weakens (Up < 50 or Down > 50) or ADX < 20
            if au < 50 or ad > 50 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Aroon weakens (Down < 50 or Up > 50) or ADX < 20
            if ad < 50 or au > 50 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals