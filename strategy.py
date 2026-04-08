#%%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's close for Camarilla (using shift)
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = close_1d[0]  # first value
    
    # Calculate Camarilla levels for each day
    # R4 = Close + (High - Low) * 1.5
    # R3 = Close + (High - Low) * 1.25
    # R2 = Close + (High - Low) * 1.166
    # R1 = Close + (High - Low) * 1.083
    # S1 = Close - (High - Low) * 1.083
    # S2 = Close - (High - Low) * 1.166
    # S3 = Close - (High - Low) * 1.25
    # S4 = Close - (High - Low) * 1.5
    
    range_1d = high_1d - low_1d
    r4 = prev_close_1d + range_1d * 1.5
    r3 = prev_close_1d + range_1d * 1.25
    r2 = prev_close_1d + range_1d * 1.166
    r1 = prev_close_1d + range_1d * 1.083
    s1 = prev_close_1d - range_1d * 1.083
    s2 = prev_close_1d - range_1d * 1.166
    s3 = prev_close_1d - range_1d * 1.25
    s4 = prev_close_1d - range_1d * 1.5
    
    # Align Camarilla levels to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d trend: 50-period EMA
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 6h volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S3 or trend fails
            if close[i] < s3_6h[i] or close[i] < ema_50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 or trend fails
            if close[i] > r3_6h[i] or close[i] > ema_50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter
            bullish = close[i] > ema_50_6h[i]
            bearish = close[i] < ema_50_6h[i]
            
            # Long: break above R4 with volume (continuation)
            if (close[i] > r4_6h[i] and 
                bullish and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: break below S4 with volume (continuation)
            elif (close[i] < s4_6h[i] and 
                  bearish and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
            # Long: bounce from S3 (mean reversion in range)
            elif (close[i] > s3_6h[i] and close[i] < s4_6h[i] and 
                  bearish and  # counter-trend bounce in bear market
                  vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: bounce from R3 (mean reversion in range)
            elif (close[i] < r3_6h[i] and close[i] > r4_6h[i] and 
                  bullish and  # counter-trend bounce in bull market
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
#%%