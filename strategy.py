#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot levels with volume confirmation and 1d ADX trend filter.
# Long when price breaks above R4 with volume > 1.5x 20-period median and 1d ADX > 25 (strong uptrend).
# Short when price breaks below S4 with volume > 1.5x 20-period median and 1d ADX > 25 (strong downtrend).
# Exit when price returns to the 12h VWAP (mean reversion to fair value).
# Uses discrete position size 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# This strategy captures strong breakouts in trending markets (ADX>25) while filtering out false breakouts in ranging markets.
# Works in both bull and bear markets by using ADX to identify strong trends regardless of direction.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Camarilla pivots and VWAP
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # === 12h Indicators: Camarilla Pivot Levels (R3, R4, S3, S4) and VWAP ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    vol_12h = df_12h['volume'].values
    
    # Calculate 12h pivot point (standard formula)
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    r3_12h = pivot_12h + range_12h * 1.1 / 4.0
    r4_12h = pivot_12h + range_12h * 1.1 / 2.0
    s3_12h = pivot_12h - range_12h * 1.1 / 4.0
    s4_12h = pivot_12h - range_12h * 1.1 / 2.0
    
    # Calculate 12h VWAP (typical price * volume)
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    vwap_12h = (typical_price_12h * vol_12h).cumsum() / vol_12h.cumsum()
    
    # Get 1d data for trend filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: ADX for trend strength filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (period=14)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period_adx = 14
    plus_dm_smooth = wilders_smoothing(plus_dm, period_adx)
    minus_dm_smooth = wilders_smoothing(minus_dm, period_adx)
    tr_smooth = wilders_smoothing(tr, period_adx)
    
    # Avoid division by zero
    plus_di = np.where(tr_smooth != 0, (plus_dm_smooth / tr_smooth) * 100, 0)
    minus_di = np.where(tr_smooth != 0, (minus_dm_smooth / tr_smooth) * 100, 0)
    dx = np.where((plus_di + minus_di) != 0, np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100, 0)
    adx = wilders_smoothing(dx, period_adx)
    
    # Get 6h volume median (20-period)
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (6h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_median_aligned = align_htf_to_ltf(prices, df_12h, vol_median_20)  # using 12h index for alignment
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20)  # ADX(30), VWAP needs history
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(r4_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or 
            np.isnan(vwap_12h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_median_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        r3 = r3_12h_aligned[i]
        r4 = r4_12h_aligned[i]
        s3 = s3_12h_aligned[i]
        s4 = s4_12h_aligned[i]
        vwap = vwap_12h_aligned[i]
        adx_val = adx_aligned[i]
        vol_median = vol_median_aligned[i]
        vol_6h = volume[i]  # current 6h volume
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position != 0:  # have a position
            # Exit when price returns to 12h VWAP (mean reversion)
            if abs(price - vwap) / vwap < 0.005:  # within 0.5% of VWAP
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume spike filter: current 6h volume > 1.5x median volume
            volume_spike = vol_6h > (vol_median * 1.5)
            # Trend filter: 1d ADX > 25 (strong trend)
            strong_trend = adx_val > 25
            
            # LONG CONDITIONS
            # Price breaks above R4 with volume spike and strong trend
            if price > r4 and volume_spike and strong_trend:
                signals[i] = 0.25
                position = 1
            
            # SHORT CONDITIONS
            # Price breaks below S4 with volume spike and strong trend
            elif price < s4 and volume_spike and strong_trend:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_VolumeSpike1.5x_ADX25_v1"
timeframe = "6h"
leverage = 1.0