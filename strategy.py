#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + 1d Trend Filter + Volume Spike
# Long when price > Kumo cloud, Tenkan > Kijun, and 1d EMA50 uptrend
# Short when price < Kumo cloud, Tenkan < Kijun, and 1d EMA50 downtrend
# Exit when price crosses back into Kumo or trend flips
# Ichimoku works in all regimes: cloud acts as dynamic S/R, TK cross signals momentum
# Trend filter prevents counter-trend trades; volume spike ensures conviction
# Target: 15-30 trades/year (60-120 over 4 years) with edge in both bull/bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Ichimoku and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): current close shifted back 26 periods
    # Not used in signals to avoid look-ahead
    
    # Kumo cloud boundaries (future Senkou spans)
    # Senkou A and B are plotted 26 periods ahead, so we use current values
    # For cloud calculation at time t, we need Senkou from t-26
    # We'll handle the shift in alignment
    
    # 1d EMA50 for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all to 6t
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        ema_50_val = ema_50_aligned[i]
        
        # Kumo cloud boundaries (Senkou A and B)
        upper_cloud = max(senkou_a_val, senkou_b_val)
        lower_cloud = min(senkou_a_val, senkou_b_val)
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price above cloud, Tenkan > Kijun, EMA50 uptrend, volume spike
            if (price > upper_cloud and 
                tenkan_val > kijun_val and 
                ema_50_val > ema_50_aligned[max(0, i-1)] and  # EMA50 rising
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short conditions: price below cloud, Tenkan < Kijun, EMA50 downtrend, volume spike
            elif (price < lower_cloud and 
                  tenkan_val < kijun_val and 
                  ema_50_val < ema_50_aligned[max(0, i-1)] and  # EMA50 falling
                  vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses into cloud or trend flips
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price drops below cloud or trend turns down
                if (price < upper_cloud or 
                    ema_50_val < ema_50_aligned[max(0, i-1)]):
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price rises above cloud or trend turns up
                if (price > lower_cloud or 
                    ema_50_val > ema_50_aligned[max(0, i-1)]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Trend_Filter_Volume"
timeframe = "6h"
leverage = 1.0