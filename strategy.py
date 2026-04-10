#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku TK Cross + 1d Cloud Filter + Volume Confirmation
# - Long when TK Cross (Tenkan/Kijun) crosses up on 6h AND price > 1d Ichimoku Cloud AND 6h volume > 1.5x 20-period average
# - Short when TK Cross crosses down on 6h AND price < 1d Ichimoku Cloud AND 6h volume > 1.5x 20-period average
# - Exit when TK Cross reverses OR price enters the 1d cloud
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Ichimoku TK Cross provides timely momentum signals
# - 1d Cloud filter ensures we trade with the higher timeframe trend bias
# - Volume confirmation reduces false breakouts

name = "6h_1d_ichimoku_tk_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 for Ichimoku (26*2)
        return np.zeros(n)
    
    # Primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 6h Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # TK Cross signals
    tk_cross_up = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))  # Crossed up
    tk_cross_down = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))  # Crossed down
    
    # Pre-compute 6h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d Ichimoku Cloud
    # 1d Tenkan-sen (9-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    # 1d Kijun-sen (26-period)
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    # 1d Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # 1d Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (period52_high_1d + period52_low_1d) / 2
    
    # The Cloud is between Senkou Span A and Senkou Span B
    # For simplicity, we'll use the current cloud values (already shifted in calculation)
    # Upper cloud = max(Senkou A, Senkou B)
    # Lower cloud = min(Senkou A, Senkou B)
    upper_cloud_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    lower_cloud_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align HTF indicators to 6h timeframe
    upper_cloud_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_cloud_1d)
    lower_cloud_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_cloud_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after warmup (max period is 52)
        # Skip if any required data is invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(upper_cloud_1d_aligned[i]) or np.isnan(lower_cloud_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Determine if price is above/below 1d cloud
        price_above_cloud = close[i] > upper_cloud_1d_aligned[i]
        price_below_cloud = close[i] < lower_cloud_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: TK Cross up AND price above 1d cloud AND volume spike
            if tk_cross_up[i] and price_above_cloud and volume_spike[i]:
                position = 1
                signals[i] = 0.25
            # Short conditions: TK Cross down AND price below 1d cloud AND volume spike
            elif tk_cross_down[i] and price_below_cloud and volume_spike[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: TK Cross reverses OR price enters the 1d cloud
            exit_long = (position == 1 and 
                        (tk_cross_down[i] or not price_above_cloud))
            exit_short = (position == -1 and 
                         (tk_cross_up[i] or not price_below_cloud))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals