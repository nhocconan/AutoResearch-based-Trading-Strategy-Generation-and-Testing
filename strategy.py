#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla Pivot R3/S3 Breakout with 1w Volume Spike and ADX Trend Filter
# Long when: price breaks above Camarilla R3 AND 1w volume > 1.3x average AND 1w ADX > 25
# Short when: price breaks below Camarilla S3 AND 1w volume > 1.3x average AND 1w ADX > 25
# Exit when price returns to Camarilla Pivot Point (mean reversion)
# Camarilla pivots identify key intraday support/resistance levels
# Volume spike confirms institutional participation
# 1w ADX > 25 ensures alignment with higher timeframe trend
# Target: 30-100 total trades over 4 years (7-25/year) with discrete sizing 0.25 to minimize fee churn

name = "1d_Camarilla_R3S3_Breakout_1wVolumeSpike_ADXTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for volume average and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for ADX and volume average
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w average volume (20-period)
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # Calculate 1w ADX(14)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0) if high[i] - high[i-1] > low[i-1] - low[i] else 0
            minus_dm[i] = max(low[i-1] - low[i], 0) if low[i-1] - low[i] > high[i] - high[i-1] else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_smoothed = np.mean(plus_dm[1:period+1])
        minus_dm_smoothed = np.mean(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smoothed = (plus_dm_smoothed * (period-1) + plus_dm[i]) / period
            minus_dm_smoothed = (minus_dm_smoothed * (period-1) + minus_dm[i]) / period
            
            plus_di[i] = 100 * plus_dm_smoothed / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_smoothed / atr[i] if atr[i] != 0 else 0
            dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
        
        # ADX is smoothed DX
        adx[2*period] = np.mean(dx[period+1:2*period+1])
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate Camarilla levels for 1d (based on previous day's OHLC)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We need to shift by 1 to avoid look-ahead (use previous day's OHLC)
    camarilla_pp = np.zeros(n)
    camarilla_r3 = np.zeros(n)
    camarilla_s3 = np.zeros(n)
    
    for i in range(1, n):
        # Use previous day's OHLC to calculate today's Camarilla levels
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        camarilla_pp[i] = (prev_high + prev_low + prev_close) / 3
        camarilla_r3[i] = prev_close + ((prev_high - prev_low) * 1.1 / 4)
        camarilla_s3[i] = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(vol_ma_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(camarilla_pp[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 + volume spike + ADX > 25
            if (close[i] > camarilla_r3[i] and 
                volume[i] > 1.3 * vol_ma_aligned[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 + volume spike + ADX > 25
            elif (close[i] < camarilla_s3[i] and 
                  volume[i] > 1.3 * vol_ma_aligned[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to pivot point (mean reversion)
            if close[i] < camarilla_pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to pivot point (mean reversion)
            if close[i] > camarilla_pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals