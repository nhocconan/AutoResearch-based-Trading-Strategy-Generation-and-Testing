#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter (EMA34) and volume confirmation
# Long when price breaks above R3 AND 1d EMA34 uptrend AND volume spike (>2.0x 20-bar MA)
# Short when price breaks below S3 AND 1d EMA34 downtrend AND volume spike
# Camarilla pivot levels provide high-probability intraday support/resistance
# 1d EMA34 filter ensures we only trade in the direction of the daily trend
# Volume spike confirms institutional participation and reduces false breakouts
# Timeframe: 12h (primary timeframe as required)
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag and maximize edge

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    # Camarilla levels: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low)
    #                S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    # We use the previous bar's high/low/close to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar: use current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Calculate Camarilla levels for each bar (based on previous bar)
    range_hl = prev_high - prev_low
    camarilla_r3 = prev_close + 1.125 * range_hl
    camarilla_s3 = prev_close - 1.125 * range_hl
    
    # Volume confirmation on 12h (threshold: 2.0x 20-bar MA)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND 1d EMA34 uptrend AND volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_34_1d_aligned[i] and  # Uptrend: price above EMA34
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND 1d EMA34 downtrend AND volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_34_1d_aligned[i] and  # Downtrend: price below EMA34
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR 1d trend turns down
            if close[i] < camarilla_s3[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 OR 1d trend turns up
            if close[i] > camarilla_r3[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals