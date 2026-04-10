#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# - Primary: 6h timeframe for balanced trade frequency (~15-35/year) and reduced fee drag
# - HTF: 1d for Ichimoku cloud calculation (Senkou Span A/B) and trend direction
# - Long: Price above 6h cloud + Tenkan > Kijun (bullish TK cross) + 1d price > Senkou Span A (1d bullish bias) + volume > 1.5x 20-period MA
# - Short: Price below 6h cloud + Tenkan < Kijun (bearish TK cross) + 1d price < Senkou Span A (1d bearish bias) + volume > 1.5x 20-period MA
# - Exit: Price crosses opposite Tenkan/Kijun line or closes outside cloud in opposite direction
# - Position sizing: 0.25 (discrete level)
# - Target: 60-120 total trades over 4 years (15-30/year) - within 6h sweet spot
# - Works in bull/bear: Ichimoku cloud acts as dynamic support/resistance; TK cross captures momentum; 1d filter avoids counter-trend trades

name = "6h_1d_ichimoku_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for Ichimoku calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 6h Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for entry)
    
    # Calculate 1d Ichimoku cloud for trend filter
    # 1d Tenkan-sen
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    # 1d Kijun-sen
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    # 1d Senkou Span A
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    
    # Calculate 6h volume moving average (20-period) for volume confirmation
    volume_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after warmup period (max lookback is 52)
        # Skip if any required data is invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(volume_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Determine 6h cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_a[i], senkou_b[i])
        lower_cloud = np.minimum(senkou_a[i], senkou_b[i])
        
        # Volume confirmation: current 6h volume > 1.5x 20-period MA
        volume_spike = volume_6h[i] > 1.5 * volume_ma_20_6h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price above cloud + bullish TK cross + 1d bullish bias + volume spike
            if (close_6h[i] > upper_cloud[i] and  # Price above cloud
                tenkan[i] > kijun[i] and          # Bullish TK cross
                close_1d[i // 24] > senkou_a_1d_aligned[i] and  # 1d price > Senkou Span A (1d bullish)
                volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below cloud + bearish TK cross + 1d bearish bias + volume spike
            elif (close_6h[i] < lower_cloud[i] and   # Price below cloud
                  tenkan[i] < kijun[i] and           # Bearish TK cross
                  close_1d[i // 24] < senkou_a_1d_aligned[i] and  # 1d price < Senkou Span A (1d bearish)
                  volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price crosses opposite Tenkan/Kijun line (TK cross reversal)
            # 2. Price closes outside cloud in opposite direction
            
            if position == 1:  # Long position
                exit_condition = (
                    tenkan[i] < kijun[i] or          # Bearish TK cross (exit long)
                    close_6h[i] < lower_cloud[i]     # Price below cloud (exit long)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    tenkan[i] > kijun[i] or          # Bullish TK cross (exit short)
                    close_6h[i] > upper_cloud[i]     # Price above cloud (exit short)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals