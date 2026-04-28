#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price action tends to respect 12h Ichimoku Cloud as dynamic support/resistance
# In bull markets, price stays above cloud (Tenkan > Kijun > Senkou Span A/B)
# In bear markets, price stays below cloud (Tenkan < Kijun < Senkou Span A/B)
# Cloud acts as filter: only trade in direction of cloud color
# Entry: Tenkan-Kijun cross in direction of cloud with volume confirmation
# Exit: Opposite cross or price re-enters cloud
# Target: 50-150 trades over 4 years (12-37/year) with 0.25 position size

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Ichimoku calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Ichimoku parameters: Tenkan=9, Kijun=26, Senkou=52
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    highest_high_9 = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    lowest_low_9 = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan = (highest_high_9 + lowest_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    highest_high_26 = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    lowest_low_26 = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun = (highest_high_26 + lowest_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    highest_high_52 = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    lowest_low_52 = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_b = ((highest_high_52 + lowest_low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe (wait for 12h bar to close)
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_b)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > vol_ma_20 * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud color and position
        # Green cloud (bullish): Senkou A > Senkou B
        # Red cloud (bearish): Senkou A < Senkou B
        green_cloud = senkou_a_aligned[i] > senkou_b_aligned[i]
        red_cloud = senkou_a_aligned[i] < senkou_b_aligned[i]
        
        # Price above/below cloud
        above_cloud = close[i] > senkou_a_aligned[i] and close[i] > senkou_b_aligned[i]
        below_cloud = close[i] < senkou_a_aligned[i] and close[i] < senkou_b_aligned[i]
        
        # Tenkan-Kijun cross
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i]
        
        # Entry conditions: TK cross in direction of cloud with volume confirmation
        long_entry = tk_cross_up and green_cloud and above_cloud and volume_confirm[i]
        short_entry = tk_cross_down and red_cloud and below_cloud and volume_confirm[i]
        
        # Exit conditions: opposite TK cross or price re-enters cloud
        if position == 1:
            exit_condition = tk_cross_down or not above_cloud
        elif position == -1:
            exit_condition = tk_cross_up or not below_cloud
        else:
            exit_condition = False
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif exit_condition and position != 0:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_IchimokuCloud_TKCross_VolumeConfirm"
timeframe = "6h"
leverage = 1.0