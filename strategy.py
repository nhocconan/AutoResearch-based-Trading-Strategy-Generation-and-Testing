#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 12h trend filter and volume confirmation
# Long when Tenkan > Kijun, price above Kumo (cloud), and Tenkan/Kijun both rising
# Short when Tenkan < Kijun, price below Kumo, and Tenkan/Kijun both falling
# Uses volume > 30-period average to confirm signals
# Ichimoku provides dynamic support/resistance and trend direction
# Target: 50-150 total trades over 4 years with balanced performance in bull/bear markets

name = "6h_ichimoku_12h_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for Ichimoku calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:  # Need enough data for Ichimoku (26*2)
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_b)
    
    # Kumo (Cloud) boundaries: Senkou Span A and B shifted forward 26 periods
    # For Ichimoku, the cloud is plotted 26 periods ahead
    # We need to shift Senkou spans back by 26 to get current cloud
    senkou_a_shifted = np.roll(senkou_a_aligned, 26)
    senkou_b_shifted = np.roll(senkou_b_aligned, 26)
    # Set first 26 values to NaN since they don't have valid cloud data
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(52, n):  # Start after Ichimoku calculation period
        # Skip if required data not available
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price falls below cloud or Tenkan crosses below Kijun
            elif close[i] < cloud_bottom[i] or tenkan_aligned[i] < kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price rises above cloud or Tenkan crosses above Kijun
            elif close[i] > cloud_top[i] or tenkan_aligned[i] > kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if vol_filter[i]:
                # Bullish: Tenkan > Kijun, price above cloud, both rising
                if (tenkan_aligned[i] > kijun_aligned[i] and 
                    close[i] > cloud_top[i] and
                    tenkan_aligned[i] > tenkan_aligned[i-1] and
                    kijun_aligned[i] > kijun_aligned[i-1]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Bearish: Tenkan < Kijun, price below cloud, both falling
                elif (tenkan_aligned[i] < kijun_aligned[i] and 
                      close[i] < cloud_bottom[i] and
                      tenkan_aligned[i] < tenkan_aligned[i-1] and
                      kijun_aligned[i] < kijun_aligned[i-1]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals