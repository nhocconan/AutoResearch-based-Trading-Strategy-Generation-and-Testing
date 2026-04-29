#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Uses Ichimoku components (Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span) from 6h
# Trend filter: price above/below 1d EMA50
# Volume confirmation: current volume > 1.5x 20-bar average
# Long: price > Senkou Span A AND Tenkan-sen > Kijun-sen AND price > 1d EMA50 AND volume spike
# Short: price < Senkou Span B AND Tenkan-sen < Kijun-sen AND price < 1d EMA50 AND volume spike
# Exit: price crosses Tenkan-sen/Kijun-sen midpoint OR price crosses 1d EMA50
# Discrete position sizing: 0.25 for long/short, 0.0 for flat to minimize fee churn
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe

name = "6h_Ichimoku_Cloud_Breakout_1dEMA50_VolumeSpike_v1"
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
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Ichimoku components (6h timeframe)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0)
    # Note: We don't actually shift forward here because align_htf_to_ltf handles timing
    # For cloud calculation, we use current values but interpret as future cloud
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (max_high_senkou_b + min_low_senkou_b) / 2.0
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind
    # Not used in this strategy as it requires looking back
    
    # Cloud top and bottom (for current price comparison)
    cloud_top = np.maximum(senkou_span_a, senkou_span_b)
    cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # Tenkan/Kijun midpoint for exit
    tk_midpoint = (tenkan_sen + kijun_sen) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(52, 20, 50)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_cloud_top = cloud_top[i]
        curr_cloud_bottom = cloud_bottom[i]
        curr_tenkan = tenkan_sen[i]
        curr_kijun = kijun_sen[i]
        curr_tk_mid = tk_midpoint[i]
        
        # Volume spike confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions: Close below TK midpoint OR price below 1d EMA50
            if curr_close < curr_tk_mid or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Close above TK midpoint OR price above 1d EMA50
            if curr_close > curr_tk_mid or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price > cloud top AND Tenkan > Kijun AND price > 1d EMA50 AND volume spike
            if (curr_close > curr_cloud_top and 
                curr_tenkan > curr_kijun and
                curr_close > curr_ema_1d and
                vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price < cloud bottom AND Tenkan < Kijun AND price < 1d EMA50 AND volume spike
            elif (curr_close < curr_cloud_bottom and 
                  curr_tenkan < curr_kijun and
                  curr_close < curr_ema_1d and
                  vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals