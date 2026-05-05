#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 AND 1d close > 1d EMA34 AND volume > 1.8x 20-period average
# Short when price breaks below Camarilla S3 AND 1d close < 1d EMA34 AND volume > 1.8x 20-period average
# Exit when price crosses 1d EMA34 (trend reversal)
# Uses 12h primary timeframe with 1d HTF for trend filter and Camarilla structure
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Camarilla levels provide support/resistance; 1d EMA34 filters for higher-timeframe trend; volume confirms participation
# No session filter to maximize opportunities on 12h timeframe

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume"
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
    
    # Get 1d data ONCE before loop for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels on 1d data (based on previous day's OHLC)
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 for each 1d bar
    # R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    camarilla_r3_1d = c_1d + 1.1 * (h_1d - l_1d)
    camarilla_s3_1d = c_1d - 1.1 * (h_1d - l_1d)
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Volume confirmation: volume > 1.8x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.8 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_1d_aligned[i]) or 
            np.isnan(camarilla_s3_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 1d close > 1d EMA34 AND volume spike
            if (close[i] > camarilla_r3_1d_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 1d close < 1d EMA34 AND volume spike
            elif (close[i] < camarilla_s3_1d_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA34 (trend reversal)
            if close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d EMA34 (trend reversal)
            if close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals