#!/usr/bin/env python3
# 6h_1d_12h_engulfing_volume_v1
# Hypothesis: Engulfing candle patterns on 6h with 12h trend filter and volume confirmation.
# In 12h uptrend: bullish engulfing + volume spike → long
# In 12h downtrend: bearish engulfing + volume spike → short
# Uses volume > 1.5x 20-period average for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull/bear by following 12h trend direction.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_12h_engulfing_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        # Bullish engulfing: current candle engulfs previous bearish candle
        bullish_engulf = (close[i] > open_price[i-1] and 
                         open_price[i] < close[i-1] and
                         close[i-1] < open_price[i-1])  # Previous candle bearish
        
        # Bearish engulfing: current candle engulfs previous bullish candle
        bearish_engulf = (open_price[i] > close[i-1] and 
                         close[i] < open_price[i-1] and
                         close[i-1] > open_price[i-1])  # Previous candle bullish
        
        if position == 1:  # Long position
            # Exit: bearish engulfing or 12h trend breaks (price < EMA34)
            if bearish_engulf or close[i] < ema34_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish engulfing or 12h trend breaks (price > EMA34)
            if bullish_engulf or close[i] > ema34_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: bullish engulfing + volume surge + 12h uptrend
            if (bullish_engulf and vol_surge and 
                close[i] > ema34_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: bearish engulfing + volume surge + 12h downtrend
            elif (bearish_engulf and vol_surge and 
                  close[i] < ema34_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals