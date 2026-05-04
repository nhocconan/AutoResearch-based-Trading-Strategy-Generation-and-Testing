#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels provide high-probability intraday support/resistance. 
# Breaking R3 (resistance 3) or S3 (support 3) with volume confirms institutional interest.
# 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend entries.
# Volume spike (>2x 20 EMA) filters breakouts with weak participation.
# Discrete sizing 0.25 limits risk. Target: 50-150 trades over 4 years (12-37/year).

name = "12h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe (completed 1d bar only)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels for 12h timeframe using previous bar's OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), etc.
    # We use the previous completed bar to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # First bar has no previous
    
    # Calculate Camarilla levels
    rang = prev_high - prev_low
    camarilla_r3 = prev_close + 1.125 * rang
    camarilla_s3 = prev_close - 1.125 * rang
    camarilla_r4 = prev_close + 1.5 * rang
    camarilla_s4 = prev_close - 1.5 * rang
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + uptrend + volume spike
            if close[i] > camarilla_r3[i] and close[i] > ema34_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 + downtrend + volume spike
            elif close[i] < camarilla_s3[i] and close[i] < ema34_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla R3/S3 midpoint OR trend changes OR volume drops
            midpoint = (camarilla_r3[i] + camarilla_s3[i]) / 2.0
            if (close[i] < midpoint or 
                close[i] < ema34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla R3/S3 midpoint OR trend changes OR volume drops
            midpoint = (camarilla_r3[i] + camarilla_s3[i]) / 2.0
            if (close[i] > midpoint or 
                close[i] > ema34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals