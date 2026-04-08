#!/usr/bin/env python3
# 6h_1d_ema_trend_follow_v1
# Hypothesis: 6-hour EMA trend following with 1-day EMA filter and volume confirmation.
# Long when price > 20 EMA and 20 EMA > 50 EMA on 6h, with price > 20 EMA on 1d and volume > 1.5x average.
# Short when price < 20 EMA and 20 EMA < 50 EMA on 6h, with price < 20 EMA on 1d and volume > 1.5x average.
# Exit when trend reverses (20 EMA crosses 50 EMA) or volume drops below average.
# Designed to capture strong trends while avoiding choppy markets, targeting 15-30 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ema_trend_follow_v1"
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
    
    # Calculate EMAs on 6h data
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).values
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).values
    
    # Calculate average volume for confirmation
    vol_series = pd.Series(volume)
    avg_volume = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).values
    
    # Align 1d EMA to 6h timeframe
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(ema_20[i]) or np.isnan(ema_50[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(ema_20_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 1:  # Long
            # Exit: trend reversal or low volume
            if ema_20[i] < ema_50[i] or vol < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: trend reversal or low volume
            if ema_20[i] > ema_50[i] or vol < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: EMA alignment with volume confirmation
            # Bullish: price > EMA20 > EMA50 on 6h, price > EMA20 on 1d, volume > 1.5x average
            if (price > ema_20[i] and ema_20[i] > ema_50[i] and 
                price > ema_20_1d_aligned[i] and vol > 1.5 * avg_volume[i]):
                position = 1
                signals[i] = 0.25
            # Bearish: price < EMA20 < EMA50 on 6h, price < EMA20 on 1d, volume > 1.5x average
            elif (price < ema_20[i] and ema_20[i] < ema_50[i] and 
                  price < ema_20_1d_aligned[i] and vol > 1.5 * avg_volume[i]):
                position = -1
                signals[i] = -0.25
    
    return signals