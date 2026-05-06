#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Ichimoku cloud + TK cross + volume confirmation
# - Uses 12h Ichimoku (9,26,52) for trend direction and cloud filter
# - Uses 6h TK cross (Tenkan/Kijun) for entry timing
# - Uses 6h volume spike for entry confirmation
# - Enters long when price above cloud + TK cross bullish + volume spike
# - Enters short when price below cloud + TK cross bearish + volume spike
# - Exits when price crosses opposite TK line or enters cloud
# - Designed to capture trends with institutional support/resistance levels
# - Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "6h_12hIchimoku_TK_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Ichimoku
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Ichimoku components (9,26,52)
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen = (pd.Series(high_12h).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_12h).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen = (pd.Series(high_12h).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_12h).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods
    senkou_b = ((pd.Series(high_12h).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low_12h).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    # Chikou Span (Lagging Span): close shifted -26 periods (not used for signals)
    
    # Convert to numpy arrays
    tenkan_sen = tenkan_sen.values
    kijun_sen = kijun_sen.values
    senkou_a = senkou_a.values
    senkou_b = senkou_b.values
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_6h = align_htf_to_ltf(prices, df_12h, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_12h, kijun_sen)
    senkou_a_6h = align_htf_to_ltf(prices, df_12h, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_12h, senkou_b)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # Volume filter (6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above cloud + TK cross bullish + volume spike
            if (close[i] > cloud_top[i] and 
                tenkan_sen_6h[i] > kijun_sen_6h[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK cross bearish + volume spike
            elif (close[i] < cloud_bottom[i] and 
                  tenkan_sen_6h[i] < kijun_sen_6h[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TK cross bearish OR price enters cloud
            if (tenkan_sen_6h[i] < kijun_sen_6h[i] or 
                close[i] < cloud_top[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TK cross bullish OR price enters cloud
            if (tenkan_sen_6h[i] > kijun_sen_6h[i] or 
                close[i] > cloud_bottom[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals