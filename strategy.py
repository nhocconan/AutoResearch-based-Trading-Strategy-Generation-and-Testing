#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Ichimoku Cloud + 1d Trend Filter + Volume Spike
# Hypothesis: Ichimoku provides robust trend and momentum signals. Combined with
# 1d trend filter and volume spikes, it captures strong momentum moves while
# avoiding counter-trend trades. Works in bull markets via bullish TK cross
# above cloud + uptrend, in bear via bearish TK cross below cloud + downtrend.
# Volume spikes confirm institutional participation.
# Target: 12-37 trades/year (50-150 total over 4 years) for 6h timeframe.

name = "6h_ichimoku_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).mean().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).mean().values
    tenkan_sen = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).mean().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).mean().values
    kijun_sen = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).mean().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).mean().values
    senkou_span_b = (high_52 + low_52) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods back
    # For signal generation, we use current close vs Senkou Span from 26 periods ago
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a, senkou_span_b)
    cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # Volume confirmation: volume > 1.8x 20-period average (moderate threshold)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(tenkan_sen[i]) or 
            np.isnan(kijun_sen[i]) or np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price falls below cloud base OR trend turns bearish
            if close[i] < cloud_bottom[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price rises above cloud top OR trend turns bullish
            if close[i] > cloud_top[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Bullish TK cross above cloud + uptrend
                if (tenkan_sen[i] > kijun_sen[i] and 
                    close[i] > cloud_top[i] and 
                    close[i] > ema_50_1d_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Bearish TK cross below cloud + downtrend
                elif (tenkan_sen[i] < kijun_sen[i] and 
                      close[i] < cloud_bottom[i] and 
                      close[i] < ema_50_1d_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals