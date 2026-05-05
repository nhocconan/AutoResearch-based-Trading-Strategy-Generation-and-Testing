#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout + 1d EMA34 trend filter + volume spike
# Long when: price > Camarilla R3 AND 1d EMA34 rising AND volume > 2.0x 20-period MA
# Short when: price < Camarilla S3 AND 1d EMA34 falling AND volume > 2.0x 20-period MA
# Exit when: price crosses Camarilla H3/L3 (mean reversion) OR volume drops below average
# Uses Camarilla for institutional pivot structure, EMA for trend direction, volume for conviction
# Timeframe: 12h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Calculate Camarilla levels on 12h (based on previous bar's range)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    for i in range(1, n):
        if not (np.isnan(high[i-1]) or np.isnan(low[i-1]) or np.isnan(close[i-1])):
            rng = high[i-1] - low[i-1]
            camarilla_r3[i] = close[i-1] + rng * 1.1/4
            camarilla_s3[i] = close[i-1] - rng * 1.1/4
            camarilla_h3[i] = close[i-1] + rng * 1.1/6
            camarilla_l3[i] = close[i-1] - rng * 1.1/6
            camarilla_h4[i] = close[i-1] + rng * 1.1/2
            camarilla_l4[i] = close[i-1] - rng * 1.1/2
    
    # Get 1d data ONCE before loop for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:  # need sufficient data for EMA
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d
    close_1d = df_1d['close'].values
    if len(close_1d) >= 34:
        ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
        # EMA rising/falling
        ema_rising = np.zeros(len(ema_34), dtype=bool)
        ema_falling = np.zeros(len(ema_34), dtype=bool)
        for i in range(1, len(ema_34)):
            if not np.isnan(ema_34[i]) and not np.isnan(ema_34[i-1]):
                ema_rising[i] = ema_34[i] > ema_34[i-1]
                ema_falling[i] = ema_34[i] < ema_34[i-1]
    else:
        ema_34 = np.full(len(close_1d), np.nan)
        ema_rising = np.zeros(len(close_1d), dtype=bool)
        ema_falling = np.zeros(len(close_1d), dtype=bool)
    
    # Align 1d EMA to 12h timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling.astype(float))
    
    # Volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above R3 AND EMA rising AND volume spike
            if (close[i] > camarilla_r3[i] and 
                ema_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S3 AND EMA falling AND volume spike
            elif (close[i] < camarilla_s3[i] and 
                  ema_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below H3 (mean reversion) OR volume drops
            if (close[i] < camarilla_h3[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above L3 (mean reversion) OR volume drops
            if (close[i] > camarilla_l3[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals