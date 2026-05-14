#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R4/S4 breakout with 12h trend filter and volume confirmation
# Long when price breaks above Camarilla R4 AND 12h close > 12h EMA50 AND volume > 2x 20-period average
# Short when price breaks below Camarilla S4 AND 12h close < 12h EMA50 AND volume > 2x 20-period average
# Exit when price crosses 12h EMA50 (trend reversal)
# Uses 6h primary timeframe with 12h HTF for trend filter and Camarilla structure
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Camarilla R4/S4 represent strong breakout levels; 12h EMA50 filters for higher-timeframe trend; volume confirms institutional participation

name = "6h_Camarilla_R4S4_Breakout_12hEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 12h data ONCE before loop for trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels on 12h data (based on previous bar's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), S4 = close - 1.5*(high-low)
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Calculate Camarilla R4 and S4 for each 12h bar
    camarilla_r4_12h = c_12h + 1.5 * (h_12h - l_12h)
    camarilla_s4_12h = c_12h - 1.5 * (h_12h - l_12h)
    
    # Align Camarilla levels to 6h timeframe (wait for 12h bar to close)
    camarilla_r4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4_12h)
    camarilla_s4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4_12h)
    
    # Volume confirmation: volume > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r4_12h_aligned[i]) or 
            np.isnan(camarilla_s4_12h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R4 AND 12h close > 12h EMA50 AND volume spike
            if (close[i] > camarilla_r4_12h_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S4 AND 12h close < 12h EMA50 AND volume spike
            elif (close[i] < camarilla_s4_12h_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 12h EMA50 (trend reversal)
            if close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 12h EMA50 (trend reversal)
            if close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals