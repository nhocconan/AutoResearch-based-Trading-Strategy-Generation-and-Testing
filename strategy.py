#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Trend + Volume Spike
# Williams Alligator (jaw=13, teeth=8, lips=5) identifies trend via aligned SMAs.
# Long when lips > teeth > jaw + volume > 2x 20-period EMA volume + 1d EMA50 uptrend.
# Short when lips < teeth < jaw + volume confirmation + 1d EMA50 downtrend.
# Designed for 12h timeframe targeting 50-150 total trades over 4 years (12-37/year).
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.
# Alligator works in both bull (trending) and bear (avoids chop via alignment) markets.

name = "12h_WilliamsAlligator_1dEMA50_Trend_Volume"
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
    open_ = prices['open'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h timeframe (SMAs of median price)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # 8-period SMA
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values     # 5-period SMA
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: lips > teeth > jaw (Alligator bullish alignment) + volume + 1d EMA50 uptrend
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                volume_confirm and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: lips < teeth < jaw (Alligator bearish alignment) + volume + 1d EMA50 downtrend
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  volume_confirm and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks (lips <= teeth OR teeth <= jaw) OR 1d EMA50 turns down
            if (lips[i] <= teeth[i] or teeth[i] <= jaw[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks (lips >= teeth OR teeth >= jaw) OR 1d EMA50 turns up
            if (lips[i] >= teeth[i] or teeth[i] >= jaw[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals