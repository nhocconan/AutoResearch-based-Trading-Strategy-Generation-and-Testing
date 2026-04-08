#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for trend and Donchian
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h Donchian breakout (20-period)
    donch_high = np.full(len(high_12h), np.nan)
    donch_low = np.full(len(low_12h), np.nan)
    for i in range(20, len(high_12h)):
        donch_high[i] = np.max(high_12h[i-20:i])
        donch_low[i] = np.min(low_12h[i-20:i])
    
    # 12h EMA trend (34-period)
    ema_34 = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align indicators to 4h timeframe
    donch_high_4h = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_4h = align_htf_to_ltf(prices, df_12h, donch_low)
    ema_34_4h = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # Volume filter: volume > 1.4x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > (vol_ma * 1.4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(34, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_4h[i]) or np.isnan(donch_low_4h[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < Donchian low or trend fails
            if close[i] < donch_low_4h[i] or close[i] < ema_34_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high or trend fails
            if close[i] > donch_high_4h[i] or close[i] > ema_34_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter
            bullish = close[i] > ema_34_4h[i]
            bearish = close[i] < ema_34_4h[i]
            
            # Long: price > Donchian high + bullish trend + volume
            if (close[i] > donch_high_4h[i] and 
                bullish and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price < Donchian low + bearish trend + volume
            elif (close[i] < donch_low_4h[i] and 
                  bearish and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals