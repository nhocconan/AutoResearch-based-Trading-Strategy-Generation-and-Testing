#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Chande Momentum Oscillator (CMO) with 1d ADX(25) trend filter and volume spike
# CMO measures momentum strength: (sum of gains - sum of losses) / (sum of gains + sum of losses)
# Long when CMO > 0 and trending up; Short when CMO < 0 and trending down
# Uses 1d ADX(25) to confirm trend strength, volume spike (>1.8x 20-bar avg) for confirmation
# Works in bull/bear: captures momentum in trending markets, avoids chop via ADX filter
# Discrete sizing 0.25; target ~100 total trades over 4 years (25/year)

name = "4h_CMO14_1dADX25_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(25) trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    atr_1d = wilder_smooth(tr, 25)
    dm_plus_smooth = wilder_smooth(dm_plus, 25)
    dm_minus_smooth = wilder_smooth(dm_minus, 25)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilder_smooth(dx, 25)
    
    # Calculate CMO(14) for 4h timeframe
    def calculate_cmo(close_arr, period):
        delta = np.diff(close_arr, prepend=close_arr[0])
        gains = np.where(delta >= 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing for gains and losses
        def smooth_wilder(arr, p):
            res = np.full_like(arr, np.nan)
            a = 1.0 / p
            if len(arr) >= p:
                res[p-1] = np.nanmean(arr[:p])
                for i in range(p, len(arr)):
                    res[i] = res[i-1] + a * (arr[i] - res[i-1])
            return res
        
        avg_gain = smooth_wilder(gains, period)
        avg_loss = smooth_wilder(losses, period)
        
        # CMO = 100 * (avg_gain - avg_loss) / (avg_gain + avg_loss)
        denom = avg_gain + avg_loss
        cmo = np.where(denom != 0, 100 * (avg_gain - avg_loss) / denom, 0)
        return cmo
    
    cmo_14 = calculate_cmo(close, 14)
    
    # Calculate volume spike filter (>1.8x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_20)
    
    # Align HTF indicators to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(cmo_14[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: positive CMO (bullish momentum) AND strong trend (ADX > 25) AND volume spike
            if cmo_14[i] > 0 and adx_1d_aligned[i] > 25 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: negative CMO (bearish momentum) AND strong trend (ADX > 25) AND volume spike
            elif cmo_14[i] < 0 and adx_1d_aligned[i] > 25 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: CMO turns negative (momentum shifts bearish)
            if cmo_14[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: CMO turns positive (momentum shifts bullish)
            if cmo_14[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals