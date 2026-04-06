#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with daily EMA trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) + price > daily EMA50 + volume > 1.5x average
# Short when Williams %R > -20 (overbought) + price < daily EMA50 + volume > 1.5x average
# Exit when Williams %R crosses -50 or price crosses daily EMA50
# Williams %R is effective in ranging markets while EMA filter avoids counter-trend trades
# Targets 75-150 total trades over 4 years (19-38/year) to avoid fee drag

name = "6h_williamsr_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Daily EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(williams_r[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: Williams %R crosses -50 or price crosses daily EMA50
        if position == 1:  # long position
            if williams_r[i] >= -50 or close[i] <= ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if williams_r[i] <= -50 or close[i] >= ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for mean reversion entries with trend and volume confirmation
            # Bullish: oversold + price above daily EMA + volume
            if (williams_r[i] < -80 and 
                close[i] > ema_1d_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Bearish: overbought + price below daily EMA + volume
            elif (williams_r[i] > -20 and 
                  close[i] < ema_1d_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals