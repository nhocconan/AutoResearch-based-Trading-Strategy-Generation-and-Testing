# 4h_1d_camarilla_pivot_volume_v1
# Hypothesis: 4h Camarilla pivot reversal with volume confirmation and 1d trend filter
# Long when price crosses above Camarilla H3 level + volume > 1.8x average + 1d trend up
# Short when price crosses below Camarilla L3 level + volume > 1.8x average + 1d trend down
# Exit when price returns to Camarilla Pivot level or trend reverses
# Designed for 20-50 trades/year on 4h timeframe with mean reversion in range and trend following in trend
# Camarilla levels work well in both bull (trend continuation) and bear (mean reversion) markets

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    # H4 = Close + 1.5*(High-Low), H3 = Close + 1.0*(High-Low), etc.
    # L4 = Close - 1.5*(High-Low), L3 = Close - 1.0*(High-Low), etc.
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for each day
    camarilla_h3 = prev_close + 1.0 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.0 * (prev_high - prev_low)
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_filter = volume[i] > 1.8 * vol_ma_20[i]
        
        # Trend filter: price relative to 1d EMA50
        is_uptrend = close[i] > ema_50_1d_aligned[i]
        is_downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions: Camarilla H3/L3 break with volume and trend
        long_entry = (close[i] > camarilla_h3_aligned[i]) and volume_filter and is_uptrend
        short_entry = (close[i] < camarilla_l3_aligned[i]) and volume_filter and is_downtrend
        
        # Exit conditions: return to Camarilla Pivot or trend reversal
        long_exit = (close[i] < camarilla_pivot_aligned[i]) or (not is_uptrend)
        short_exit = (close[i] > camarilla_pivot_aligned[i]) or (not is_downtrend)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals