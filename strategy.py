#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and 1d ADX trend filter
# - Long when price breaks above Camarilla H3 level AND 12h volume > 1.3x 20-period average AND 1d ADX > 25 (trending)
# - Short when price breaks below Camarilla L3 level AND 12h volume > 1.3x 20-period average AND 1d ADX > 25 (trending)
# - Exit when price returns to Camarilla Pivot point
# - Uses discrete position sizing 0.25 to limit fee churn
# - Camarilla levels provide intraday support/resistance structure; volume confirms breakout validity
# - Daily ADX filter ensures we trade only in trending markets (avoids chop)
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_12h_1d_camarilla_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h Camarilla levels (based on previous day's OHLC)
    def calculate_camarilla(h_prev, l_prev, c_prev):
        range_val = h_prev - l_prev
        if range_val <= 0:
            return c_prev, c_prev, c_prev, c_prev, c_prev, c_prev, c_prev, c_prev
        camarilla_pivot = (h_prev + l_prev + c_prev) / 3.0
        camarilla_h3 = camarilla_pivot + (range_val * 1.1 / 4.0)
        camarilla_l3 = camarilla_pivot - (range_val * 1.1 / 4.0)
        camarilla_h4 = camarilla_pivot + (range_val * 1.1 / 2.0)
        camarilla_l4 = camarilla_pivot - (range_val * 1.1 / 2.0)
        return camarilla_pivot, camarilla_h3, camarilla_l3, camarilla_h4, camarilla_l4
    
    camarilla_pivot = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    
    for i in range(1, n):
        pivot, h3, l3, _, _, _, _, _ = calculate_camarilla(high[i-1], low[i-1], close[i-1])
        camarilla_pivot[i] = pivot
        camarilla_h3[i] = h3
        camarilla_l3[i] = l3
    
    # Pre-compute 4h ATR (14-period) for stoploss
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[1:15])  # First ATR value
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Pre-compute 12h volume average (20-period)
    volume_12h = df_12h['volume'].values
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    vol_ma_12h = rolling_mean(volume_12h, 20)
    
    # Pre-compute 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    def calculate_adx(h, l, c, period=14):
        # True Range
        tr1 = h - l
        tr2 = np.abs(h - np.roll(c, 1))
        tr3 = np.abs(l - np.roll(c, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = h[0] - l[0]
        
        # Directional Movement
        dm_plus = np.where((h - np.roll(h, 1)) > (np.roll(l, 1) - l), np.maximum(h - np.roll(h, 1), 0), 0)
        dm_minus = np.where((np.roll(l, 1) - l) > (h - np.roll(h, 1)), np.maximum(np.roll(l, 1) - l, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        def smooth(arr, period):
            result = np.full_like(arr, np.nan, dtype=float)
            if len(arr) < period:
                return result
            result[period-1] = np.mean(arr[1:period])  # Skip first element for DM
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        atr_smooth = smooth(tr, period)
        dm_plus_smooth = smooth(dm_plus, period)
        dm_minus_smooth = smooth(dm_minus, period)
        
        # Directional Indicators
        di_plus = np.full_like(c, np.nan, dtype=float)
        di_minus = np.full_like(c, np.nan, dtype=float)
        dx = np.full_like(c, np.nan, dtype=float)
        
        valid = ~np.isnan(atr_smooth) & (atr_smooth != 0)
        di_plus[valid] = (dm_plus_smooth[valid] / atr_smooth[valid]) * 100
        di_minus[valid] = (dm_minus_smooth[valid] / atr_smooth[valid]) * 100
        
        dx_valid = ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0)
        dx[dx_valid] = (np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / (di_plus[dx_valid] + di_minus[dx_valid])) * 100
        
        # ADX (smoothed DX)
        adx = np.full_like(c, np.nan, dtype=float)
        if len(dx) >= period:
            first_valid = np.where(~np.isnan(dx))[0]
            if len(first_valid) > 0:
                start_idx = first_valid[0]
                if start_idx + period < len(dx):
                    adx[start_idx + period - 1] = np.mean(dx[start_idx:start_idx + period])
                    for i in range(start_idx + period, len(dx)):
                        if not np.isnan(dx[i]):
                            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align HTF indicators to 4h timeframe
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or np.isnan(camarilla_pivot[i]) or 
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume spike condition
            vol_ma_4h = rolling_mean(volume, 20)
            vol_spike = not np.isnan(vol_ma_4h[i]) and volume[i] > 1.3 * vol_ma_4h[i]
            
            # Long conditions: Camarilla H3 breakout AND volume spike AND 1d ADX > 25 (trending)
            if (close[i] > camarilla_h3[i] and vol_spike and 
                adx_1d_aligned[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short conditions: Camarilla L3 breakdown AND volume spike AND 1d ADX > 25 (trending)
            elif (close[i] < camarilla_l3[i] and vol_spike and 
                  adx_1d_aligned[i] > 25):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to Camarilla Pivot point
            exit_long = (position == 1 and close[i] <= camarilla_pivot[i])
            exit_short = (position == -1 and close[i] >= camarilla_pivot[i])
            
            # Optional: ATR-based stoploss
            stop_long = (position == 1 and close[i] <= high[i] - 2.5 * atr[i])
            stop_short = (position == -1 and close[i] >= low[i] + 2.5 * atr[i])
            
            if exit_long or exit_short or stop_long or stop_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

def rolling_mean(arr, window):
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        result[i] = np.mean(arr[i - window + 1:i + 1])
    return result