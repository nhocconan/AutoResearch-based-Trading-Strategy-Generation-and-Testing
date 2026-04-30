#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d ADX25 trend filter and volume spike confirmation
# Camarilla pivot levels provide precise support/resistance based on prior day's range
# Breakout above R3 or below S3 with volume confirmation indicates strong momentum
# 1d ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranging conditions
# Volume spike (2.0x 24-period average) confirms institutional participation
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Camarilla_R3S3_1dADX25_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR and DM (Wilder's smoothing)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(data)):
            if not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    atr = wilders_smooth(tr, 14)
    dm_plus_smooth = wilders_smooth(dm_plus, 14)
    dm_minus_smooth = wilders_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilders_smooth(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    if len(df_1d) < 2:
        return np.zeros(n)
    # Camarilla levels based on prior day's OHLC
    # R3 = C + ((H-L) * 1.25/2), S3 = C - ((H-L) * 1.25/2)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.25 / 2)
    camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.25 / 2)
    
    # Align to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 24-period average (24*12h = 12 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(30, 2)  # warmup for ADX and 1d data
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_adx = adx_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and ADX > 25 (trending market)
            if curr_volume_spike and curr_adx > 25:
                # Bullish entry: break above R3 with close > R3
                if curr_close > curr_r3:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below S3 with close < S3
                elif curr_close < curr_s3:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below R3 (breakout fails)
            if curr_close < curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above S3 (breakdown fails)
            if curr_close > curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals