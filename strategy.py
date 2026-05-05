#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Long when: price breaks above R3 AND close > 4h EMA50 AND volume > 2x 20-period MA
# Short when: price breaks below S3 AND close < 4h EMA50 AND volume > 2x 20-period MA
# Exit when: price returns to R2/S2 level OR volume drops below average
# Uses Camarilla for institutional levels, 4h EMA for trend, volume for conviction
# Timeframe: 1h, HTF: 4h for EMA. Target: 80-150 total trades over 4 years (20-37/year) to avoid fee drag.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-period volume MA on 1h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    # Calculate Camarilla levels on 1h (using previous bar's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.25*(high-low)
    #          S3 = close - 1.25*(high-low), S4 = close - 1.5*(high-low)
    #          R2 = close + 1.125*(high-low), S2 = close - 1.125*(high-low)
    #          R1 = close + 1.075*(high-low), S1 = close - 1.075*(high-low)
    #          PP = (high + low + close) / 3
    
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    R3 = prev_close + 1.25 * (prev_high - prev_low)
    S3 = prev_close - 1.25 * (prev_high - prev_low)
    R2 = prev_close + 1.125 * (prev_high - prev_low)
    S2 = prev_close - 1.125 * (prev_high - prev_low)
    
    # Get 4h data ONCE before loop for EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h
    if len(df_4h) >= 50:
        ema_50 = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    else:
        ema_50 = np.full(len(df_4h), np.nan)
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(R2[i]) or np.isnan(S2[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND above 4h EMA50 AND volume spike
            if (close[i] > R3[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S3 AND below 4h EMA50 AND volume spike
            elif (close[i] < S3[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to R2 level OR volume drops below average
            if (close[i] <= R2[i] or not volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to S2 level OR volume drops below average
            if (close[i] >= S2[i] or not volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals