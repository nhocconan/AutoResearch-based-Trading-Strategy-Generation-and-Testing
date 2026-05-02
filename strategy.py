#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and ADX regime filter
# Camarilla pivot levels provide high-probability support/resistance; breaks above R3 or below S3
# with volume confirmation indicate strong momentum. 1d ADX > 25 ensures trades only in trending
# markets to avoid false breakouts in ranging conditions. Works in bull markets (buying R3 breakouts
# in uptrend) and bear markets (selling S3 breakdowns in downtrend) by taking trades only when
# ADX confirms trend strength. Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.

name = "4h_Camarilla_R3S3_Breakout_1dADX_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+ and DM- with Wilder's smoothing (alpha = 1/period)
    period_adx = 14
    alpha = 1.0 / period_adx
    
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values (simple average)
    atr[period_adx-1] = np.mean(tr[:period_adx])
    dm_plus_smooth[period_adx-1] = np.mean(dm_plus[:period_adx])
    dm_minus_smooth[period_adx-1] = np.mean(dm_minus[:period_adx])
    
    # Wilder's smoothing
    for i in range(period_adx, len(tr)):
        atr[i] = atr[i-1] * (1 - alpha) + alpha * tr[i]
        dm_plus_smooth[i] = dm_plus_smooth[i-1] * (1 - alpha) + alpha * dm_plus[i]
        dm_minus_smooth[i] = dm_minus_smooth[i-1] * (1 - alpha) + alpha * dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.zeros_like(dx)
    adx[2*period_adx-1] = np.mean(dx[period_adx:2*period_adx])  # First ADX value
    for i in range(2*period_adx, len(dx)):
        adx[i] = adx[i-1] * (1 - alpha) + alpha * dx[i]
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1d volume MA for spike detection
    vol_ma_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Camarilla levels: based on previous day's range
    close_1d_shift = np.roll(close_1d, 1)
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    
    # First value will be NaN due to roll, handle later
    camarilla_range = high_1d_shift - low_1d_shift
    camarilla_close = close_1d_shift
    
    # R3 and S3 levels
    r3 = camarilla_close + camarilla_range * 1.1 / 4
    s3 = camarilla_close - camarilla_range * 1.1 / 4
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: 1.5x 20-period EMA (~3.3 days for 4h)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for ADX calculation)
    start_idx = 2 * period_adx + 1  # Need 2*14+1 = 29 days for ADX, plus 1 for Camarilla shift
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 with volume spike AND ADX > 25 (trending market)
            if (close[i] > r3_aligned[i] and 
                volume_spike[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 with volume spike AND ADX > 25 (trending market)
            elif (close[i] < s3_aligned[i] and 
                  volume_spike[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below S3 (reversion to mean) OR ADX < 20 (trend weakening)
            if close[i] < s3_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above R3 (reversion to mean) OR ADX < 20 (trend weakening)
            if close[i] > r3_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals