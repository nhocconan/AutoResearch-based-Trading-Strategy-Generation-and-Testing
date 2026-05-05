#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w volume spike and 1w EMA34 trend filter
# Long when price breaks above R3 AND volume > 2.0x 20-period average AND 1w EMA34 > EMA34_prev (uptrend)
# Short when price breaks below S3 AND volume > 2.0x 20-period average AND 1w EMA34 < EMA34_prev (downtrend)
# Exit when price crosses back to H3/L3 level OR 1w EMA34 flips direction
# Uses discrete sizing (0.25) to limit fee drag. Target: 15-35 trades/year per symbol.
# Camarilla levels provide key support/resistance, volume spike confirms institutional interest,
# 1w EMA34 filters for primary trend direction to avoid counter-trend whipsaws.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# 1d timeframe minimizes trade frequency to reduce fee drag while capturing multi-week trends.

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
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
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We use previous day's data to avoid look-ahead
    prev_high = np.concatenate([[np.nan], df_1d['high'].values[:-1]])
    prev_low = np.concatenate([[np.nan], df_1d['low'].values[:-1]])
    prev_close = np.concatenate([[np.nan], df_1d['close'].values[:-1]])
    
    rang = prev_high - prev_low
    camarilla_h3 = prev_close + (rang * 1.1 / 4)
    camarilla_l3 = prev_close - (rang * 1.1 / 4)
    camarilla_h4 = prev_close + (rang * 1.1 / 2)
    camarilla_l4 = prev_close - (rang * 1.1 / 2)
    
    # Align Camarilla levels to 1d timeframe (same timeframe, so direct assignment with 1-bar lag)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Get 1w data for EMA34 trend filter and volume spike
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate EMA34 on 1w close
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_prev = np.concatenate([[np.nan], ema_34[:-1]])  # Previous EMA for trend direction
    
    # Uptrend when current EMA34 > previous EMA34
    uptrend_1w = ema_34 > ema_34_prev
    downtrend_1w = ema_34 < ema_34_prev
    
    # Align 1w trend to 1d timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # Volume confirmation: 1w volume > 2.0x 20-period average (spike filter)
    if len(volume_1w) >= 20:
        vol_ma_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
        volume_filter_1w = volume_1w > (2.0 * vol_ma_20)
    else:
        volume_filter_1w = np.zeros(len(df_1w), dtype=bool)
    
    # Align 1w volume filter to 1d timeframe
    volume_filter_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_filter_1w.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or 
            np.isnan(downtrend_1w_aligned[i]) or 
            np.isnan(volume_filter_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above H3 AND volume spike AND 1w uptrend
            if (close[i] > h3_aligned[i] and 
                volume_filter_1w_aligned[i] > 0.5 and 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below L3 AND volume spike AND 1w downtrend
            elif (close[i] < l3_aligned[i] and 
                  volume_filter_1w_aligned[i] > 0.5 and 
                  downtrend_1w_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back to L3 OR 1w trend flips to downtrend
            if (close[i] < l3_aligned[i] or 
                downtrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back to H3 OR 1w trend flips to uptrend
            if (close[i] > h3_aligned[i] or 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals