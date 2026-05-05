#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R4/S4 breakout with 1d volume spike and 1w EMA34 trend filter
# Long when: price breaks above weekly R4, volume > 2x 20-period average, and close > 1w EMA34
# Short when: price breaks below weekly S4, volume > 2x 20-period average, and close < 1w EMA34
# Exit when price returns to weekly R4/S4 level (mean reversion) or opposite breakout
# Uses weekly Camarilla levels for structure, effective in both bull (breakout continuation) and bear (mean reversion via exits) markets.
# Timeframe: 6h, HTF: 1w. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Camarilla_R4S4_Breakout_1wEMA34_VolumeSpike"
timeframe = "6h"
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
    
    # Calculate volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for Camarilla levels and EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly Camarilla levels from previous 1w bar
    # R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    # Using previous week's values (shifted by 1)
    if len(high_1w) >= 2:
        prev_high = np.roll(high_1w, 1)
        prev_low = np.roll(low_1w, 1)
        prev_close = np.roll(close_1w, 1)
        prev_high[0] = np.nan  # First value has no previous
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        rang = prev_high - prev_low
        camarilla_r4 = prev_close + 1.1 * rang * 1.1 / 2
        camarilla_s4 = prev_close - 1.1 * rang * 1.1 / 2
    else:
        camarilla_r4 = np.full(len(close_1w), np.nan)
        camarilla_s4 = np.full(len(close_1w), np.nan)
    
    # Align weekly Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above weekly R4, volume filter, and above 1w EMA34
            if (close[i] > camarilla_r4_aligned[i] and 
                open_price[i] <= camarilla_r4_aligned[i] and  # Ensure breakout happens on this bar
                volume_filter[i] and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below weekly S4, volume filter, and below 1w EMA34
            elif (close[i] < camarilla_s4_aligned[i] and 
                  open_price[i] >= camarilla_s4_aligned[i] and  # Ensure breakdown happens on this bar
                  volume_filter[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below weekly R4 (mean reversion) or breaks below weekly S4 (reversal)
            if close[i] < camarilla_r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above weekly S4 (mean reversion) or breaks above weekly R4 (reversal)
            if close[i] > camarilla_s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals