#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h volume confirmation and 1d ADX trend filter.
# Long when price breaks above Camarilla R3 (1h) AND 4h volume > 1.5x 20-period average AND 1d ADX > 25 (trending).
# Short when price breaks below Camarilla S3 (1h) AND 4h volume > 1.5x 20-period average AND 1d ADX > 25 (trending).
# Uses discrete position size 0.20. Camarilla levels provide intraday support/resistance, volume confirms breakout strength,
# ADX ensures we only trade in trending markets to avoid false breakouts in ranging conditions.
# Target: 80-120 total trades over 4 years (20-30/year) to balance opportunity and fee drag for 1h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Primary timeframe: 1h
    open_1h = prices['open'].values
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    # === 1h Indicators: Camarilla Pivot Levels (based on previous day) ===
    # Calculate daily pivot from previous 1d bar (using open_time to group by day)
    # Since we cannot resample, we use the prior day's OHLC from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get previous day's OHLC for Camarilla calculation
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Align 1d OHLC to 1h timeframe
    prev_close_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    prev_high_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    
    # Camarilla levels: based on previous day's range
    R3 = prev_close_1d_aligned + 1.1 * (prev_high_1d_aligned - prev_low_1d_aligned) / 2
    S3 = prev_close_1d_aligned - 1.1 * (prev_high_1d_aligned - prev_low_1d_aligned) / 2
    
    # === 4h Indicators: Volume Spike ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume_4h > (1.5 * vol_ma_4h)
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h)
    
    # === 1d Indicators: ADX for trend filter ===
    if len(df_1d) < 15:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period_adx = 14
    atr = wilders_smoothing(tr, period_adx)
    dm_plus_smooth = wilders_smoothing(dm_plus, period_adx)
    dm_minus_smooth = wilders_smoothing(dm_minus, period_adx)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, period_adx)
    
    # Align ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(volume_spike_4h_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close_1h[i]
        r3_level = R3[i]
        s3_level = S3[i]
        vol_spike = volume_spike_4h_aligned[i]
        adx_val = adx_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below S3 or ADX weakens (< 20) or volume spike ends
            if price < s3_level or adx_val < 20 or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above R3 or ADX weakens (< 20) or volume spike ends
            if price > r3_level or adx_val < 20 or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0 and in_session:
            # LONG: Price breaks above R3 AND volume spike AND ADX > 25 (strong trend)
            if price > r3_level and vol_spike and adx_val > 25:
                signals[i] = 0.20
                position = 1
            
            # SHORT: Price breaks below S3 AND volume spike AND ADX > 25 (strong trend)
            elif price < s3_level and vol_spike and adx_val > 25:
                signals[i] = -0.20
                position = -1
        
        else:
            # Hold current position
            signals[i] = position * 0.20
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hVolume_1dADX_V1"
timeframe = "1h"
leverage = 1.0