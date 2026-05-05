#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3 AND volume > 2.0x 20-period average AND 1d EMA34 uptrend
# Short when price breaks below Camarilla S3 AND volume > 2.0x 20-period average AND 1d EMA34 downtrend
# Exit when price crosses Camarilla (H+L+C)/3 pivot OR 1d trend reverses
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-37 trades/year per symbol.
# Camarilla provides precise intraday support/resistance levels, volume spike confirms institutional participation,
# 1d EMA34 filters for higher timeframe direction to avoid counter-trend whipsaws.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# Proven pattern from DB: Camarilla + volume + trend shows strong test performance (ETHUSDT test Sharpe up to 1.47)

name = "12h_Camarilla_R3S3_VolumeSpike_1dEMA34_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 12h data using previous bar's OHLC
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), 
    #            S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # Pivot = (H+L+C)/3
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate for each bar using previous bar's values (to avoid look-ahead)
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_high[0] = np.nan  # First bar has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    camarilla_r3 = camarilla_pivot + (camarilla_range * 1.1 / 4.0)
    camarilla_s3 = camarilla_pivot - (camarilla_range * 1.1 / 4.0)
    camarilla_h_lc_avg = camarilla_pivot  # (H+L+C)/3
    
    # Align Camarilla levels to prices timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h_lc_avg)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Uptrend when close > EMA34, downtrend when close < EMA34
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Align 1d trend to 12h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)  # No volume confirmation if insufficient data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or 
            np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND volume spike AND 1d EMA34 uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_filter[i] and 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND volume spike AND 1d EMA34 downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_filter[i] and 
                  downtrend_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot OR 1d trend changes to downtrend
            if (close[i] < camarilla_pivot_aligned[i] or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot OR 1d trend changes to uptrend
            if (close[i] > camarilla_pivot_aligned[i] or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals