#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
# - Tenkan-sen (9-period) and Kijun-sen (26-period) cross generates signals
# - Senkou Span A/B form cloud; price above/below cloud confirms trend
# - 1d EMA50 filters for higher timeframe trend alignment
# - Volume > 1.5x average confirms institutional participation
# Designed for low trade frequency to minimize fee drag on 6h timeframe.
name = "6h_Ichimoku_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components (6h timeframe)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_52 + min_low_52) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods back
    # Not used for signals to avoid look-ahead
    
    # Align 1d EMA to 6s timeframe
    # Senkou spans need to be aligned properly - they are plotted 26 periods ahead
    # For signal generation at time t, we need Senkou values that were known 26 periods ago
    # So we shift Senkou spans BACK by 26 to get their actual plotting position
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # Fill the rolled values with NaN for the first 26 periods
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26)  # Need 52 for Senkou B and 26 for alignment
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_shifted[i]) or np.isnan(senkou_b_shifted[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current Ichimoku values
        tenkan_i = tenkan[i]
        kijun_i = kijun[i]
        senkou_a_i = senkou_a_shifted[i]
        senkou_b_i = senkou_b_shifted[i]
        close_i = close[i]
        ema_1d = ema_50_1d_aligned[i]
        vol = volume[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_i, senkou_b_i)
        cloud_bottom = min(senkou_a_i, senkou_b_i)
        
        # Calculate 20-period volume average for spike detection
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Price above cloud AND Tenkan > Kijun AND price > 1d EMA50 AND volume > 1.5x average
            if (close_i > cloud_top and tenkan_i > kijun_i and 
                close_i > ema_1d and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Enter short: Price below cloud AND Tenkan < Kijun AND price < 1d EMA50 AND volume > 1.5x average
            elif (close_i < cloud_bottom and tenkan_i < kijun_i and 
                  close_i < ema_1d and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price below cloud OR Tenkan < Kijun OR trend reverses (price < 1d EMA50)
            if (close_i < cloud_bottom or tenkan_i < kijun_i or close_i < ema_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price above cloud OR Tenkan > Kijun OR trend reverses (price > 1d EMA50)
            if (close_i > cloud_top or tenkan_i > kijun_i or close_i > ema_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals