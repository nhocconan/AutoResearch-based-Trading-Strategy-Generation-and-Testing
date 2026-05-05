#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h volume spike and 1d EMA34 trend filter
# Long when price breaks above R1 AND volume > 1.8x 20-period average AND 1d EMA34 > EMA34_prev (uptrend)
# Short when price breaks below S1 AND volume > 1.8x 20-period average AND 1d EMA34 < EMA34_prev (downtrend)
# Exit when price crosses back to H3/L3 level OR 1d EMA34 flips direction
# Uses discrete sizing (0.25) to limit fee drag. Target: 20-50 trades/year per symbol.
# Camarilla levels provide intraday support/resistance, volume spike confirms institutional interest,
# 1d EMA34 filters for primary trend direction to avoid counter-trend whipsaws.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "4h_Camarilla_R1S1_Breakout_12hVolumeSpike_1dEMA34_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla levels calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 1d data (using previous day's OHLC)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), R1 = C + ((H-L)*1.1/6)
    # S1 = C - ((H-L)*1.1/6), S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    prev_high = np.concatenate([[np.nan], df_1d['high'].values[:-1]])
    prev_low = np.concatenate([[np.nan], df_1d['low'].values[:-1]])
    prev_close = np.concatenate([[np.nan], df_1d['close'].values[:-1]])
    
    rang = prev_high - prev_low
    camarilla_r1 = prev_close + (rang * 1.1 / 6)
    camarilla_s1 = prev_close - (rang * 1.1 / 6)
    camarilla_h3 = prev_close + (rang * 1.1 / 4)
    camarilla_l3 = prev_close - (rang * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Get 12h data for volume spike filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    if len(volume_12h) >= 20:
        vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
        volume_spike_12h = volume_12h > (1.8 * vol_ma_20)
    else:
        volume_spike_12h = np.zeros(len(volume_12h), dtype=bool)
    
    # Align 12h volume spike to 4h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h.astype(float))
    
    # Get 1d data for EMA34 trend filter
    close_1d = df_1d['close'].values
    if len(close_1d) >= 34:
        ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
        ema_34_prev = np.concatenate([[np.nan], ema_34[:-1]])  # Previous EMA for trend direction
        
        # Uptrend when current EMA34 > previous EMA34
        uptrend_1d = ema_34 > ema_34_prev
        downtrend_1d = ema_34 < ema_34_prev
    else:
        uptrend_1d = np.zeros(len(close_1d), dtype=bool)
        downtrend_1d = np.zeros(len(close_1d), dtype=bool)
        ema_34_prev = np.array([])
    
    # Align 1d trend to 4h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or 
            np.isnan(downtrend_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R1 AND volume spike AND 1d uptrend
            if (close[i] > r1_aligned[i] and 
                volume_spike_aligned[i] > 0.5 and 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S1 AND volume spike AND 1d downtrend
            elif (close[i] < s1_aligned[i] and 
                  volume_spike_aligned[i] > 0.5 and 
                  downtrend_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back to L3 OR 1d trend flips to downtrend
            if (close[i] < l3_aligned[i] or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back to R3 OR 1d trend flips to uptrend
            if (close[i] > h3_aligned[i] or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals