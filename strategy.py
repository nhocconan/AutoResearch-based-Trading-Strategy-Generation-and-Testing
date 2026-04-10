#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and 1w ADX regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.3x 20-period average AND 1w ADX > 20 (trending)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.3x 20-period average AND 1w ADX > 20
# - Exit when price returns to Camarilla PIVOT (mean of H3 and L3) or reverses to opposite H4/L4
# - Uses discrete position sizing 0.25 to limit fee churn
# - Camarilla levels derived from 1d OHLC provide institutional support/resistance structure
# - Volume confirmation reduces false breakouts
# - Weekly ADX filter ensures we trade only when higher timeframe is trending (avoids chop)
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_1w_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 1d Camarilla levels (based on previous day OHLC)
    # Camarilla: H4 = C + 1.1*(H-L)/2, H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4, L4 = C - 1.1*(H-L)/2
    # Pivot = (H+L+C)/3
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low) / 2.0
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4.0
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4.0
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low) / 2.0
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i - 19:i + 1])
    
    # Pre-compute 1w ADX(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Wilder's smoothing function
    def wheilder_smoothing(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value: simple average
        result[period-1] = np.nansum(arr[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr_1w = wheilder_smoothing(tr, 14)
    dm_plus_smooth = wheilder_smoothing(dm_plus, 14)
    dm_minus_smooth = wheilder_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.full_like(atr_1w, np.nan, dtype=float)
    di_minus = np.full_like(atr_1w, np.nan, dtype=float)
    for i in range(14, len(atr_1w)):
        if not np.isnan(atr_1w[i]) and atr_1w[i] != 0:
            di_plus[i] = (dm_plus_smooth[i] / atr_1w[i]) * 100
            di_minus[i] = (dm_minus_smooth[i] / atr_1w[i]) * 100
    
    # DX and ADX
    dx = np.full_like(di_plus, np.nan, dtype=float)
    for i in range(14, len(di_plus)):
        if not np.isnan(di_plus[i]) and not np.isnan(di_minus[i]):
            di_sum = di_plus[i] + di_minus[i]
            if di_sum != 0:
                dx[i] = np.abs(di_plus[i] - di_minus[i]) / di_sum * 100
    
    adx_1w = wheilder_smoothing(dx, 14)
    
    # Align HTF indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(adx_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume spike condition (1.3x average)
            vol_ma_4h = np.full_like(volume, np.nan, dtype=float)
            for j in range(19, i+1):
                vol_ma_4h[j] = np.mean(volume[j-19:j+1])
            vol_spike = not np.isnan(vol_ma_4h[i]) and volume[i] > 1.3 * vol_ma_4h[i]
            
            # Long conditions: price > H3 AND volume spike AND 1w trending (ADX > 20)
            if (close[i] > camarilla_h3_aligned[i] and vol_spike and adx_1w_aligned[i] > 20):
                position = 1
                signals[i] = 0.25
            # Short conditions: price < L3 AND volume spike AND 1w trending (ADX > 20)
            elif (close[i] < camarilla_l3_aligned[i] and vol_spike and adx_1w_aligned[i] > 20):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to pivot or reverses to opposite H4/L4
            exit_long = (position == 1 and (close[i] < camarilla_pivot_aligned[i] or close[i] > camarilla_h4_aligned[i]))
            exit_short = (position == -1 and (close[i] > camarilla_pivot_aligned[i] or close[i] < camarilla_l4_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

def wheilder_smoothing(arr, period):
    result = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < period:
        return result
    # First value: simple average
    result[period-1] = np.nansum(arr[:period])
    # Subsequent values: Wilder's smoothing
    for i in range(period, len(arr)):
        if not np.isnan(result[i-1]):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
    return result