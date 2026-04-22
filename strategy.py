#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d Kijun-Sen filter and volume confirmation
# Uses Ichimoku components: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (52)
# Long when price > cloud, Tenkan > Kijun, and 1d Kijun-sen trending up + volume spike
# Short when price < cloud, Tenkan < Kijun, and 1d Kijun-sen trending down + volume spike
# Ichimoku provides dynamic support/resistance; Kijun-sen acts as dynamic equilibrium
# Designed for 6h timeframe to target 20-40 trades/year per symbol.
# Works in bull (captures trend continuation) and bear (avoids false breaks via cloud filter)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Kijun-sen (26-period) for higher timeframe trend filter
    kijun_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    kijun_1d = kijun_1d.values
    kijun_1d_slope = kijun_1d - np.roll(kijun_1d, 1)
    kijun_1d_slope[0] = 0
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    kijun_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d_slope)
    
    # Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
              pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    tenkan = tenkan.values
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
             pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    kijun = kijun.values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low).rolling(window=52, min_periods=52).min()) / 2)
    senkou_b = senkou_b.values
    
    # Current cloud boundaries (shifted back by 26 to align with current price)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # Volume spike filter (20-period on 6h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(kijun_1d_aligned[i]) or np.isnan(kijun_1d_slope_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above cloud, Tenkan > Kijun, 1d Kijun rising, volume spike
            if (close[i] > cloud_top[i] and 
                tenkan[i] > kijun[i] and 
                kijun_1d_slope_aligned[i] > 0 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud, Tenkan < Kijun, 1d Kijun falling, volume spike
            elif (close[i] < cloud_bottom[i] and 
                  tenkan[i] < kijun[i] and 
                  kijun_1d_slope_aligned[i] < 0 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price crosses cloud or Tenkan/Kijun cross
            if position == 1:
                # Exit on price below cloud or Tenkan < Kijun
                if (close[i] < cloud_bottom[i] or 
                    tenkan[i] < kijun[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on price above cloud or Tenkan > Kijun
                if (close[i] > cloud_top[i] or 
                    tenkan[i] > kijun[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_1dKijun_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0