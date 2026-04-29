#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with TK Cross and Weekly EMA50 Trend Filter
# Ichimoku provides dynamic support/resistance via cloud (Senkou Span A/B)
# TK Cross (Tenkan/Kijun) signals momentum shifts
# Weekly EMA50 ensures alignment with major trend to avoid counter-trend trades
# Volume confirmation (>1.5x 24-period average) filters low-quality signals
# Works in bull/bear: cloud acts as dynamic S/R, TK cross catches momentum, weekly trend filter avoids whipsaws
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_Ichimoku_TK_Cross_WeeklyEMA50_Trend_Volume_v1"
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
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Calculate weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5x 24-period average (4 days of 6h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52 + 26, 26, 9, 24, 50)  # warmup for Senkou B shift, Kijun, Tenkan, volume MA, weekly EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(vol_ma_24[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_tenkan = tenkan[i]
        curr_kijun = kijun[i]
        curr_senkou_a = senkou_a[i]
        curr_senkou_b = senkou_b[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50 = ema_50_aligned[i]
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = max(curr_senkou_a, curr_senkou_b)
        lower_cloud = min(curr_senkou_a, curr_senkou_b)
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: TK Cross bullish (Tenkan > Kijun) AND price above cloud AND price above weekly EMA50
                if (curr_tenkan > curr_kijun and 
                    curr_close > upper_cloud and 
                    curr_close > curr_ema_50):
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: TK Cross bearish (Tenkan < Kijun) AND price below cloud AND price below weekly EMA50
                elif (curr_tenkan < curr_kijun and 
                      curr_close < lower_cloud and 
                      curr_close < curr_ema_50):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when TK Cross turns bearish OR price falls below cloud
            if (curr_tenkan < curr_kijun) or (curr_close < lower_cloud):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when TK Cross turns bullish OR price rises above cloud
            if (curr_tenkan > curr_kijun) or (curr_close > upper_cloud):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals