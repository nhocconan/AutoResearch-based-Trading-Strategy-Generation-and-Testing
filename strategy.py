#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d/1w trend filter
# - Long when price > Kumo (cloud) AND Tenkan > Kijun (bullish TK cross) AND 1d close > 1w close (bullish weekly)
# - Short when price < Kumo AND Tenkan < Kijun (bearish TK cross) AND 1d close < 1w close (bearish weekly)
# - Exit: price re-enters Kumo or TK cross reverses
# - Uses Ichimoku for trend/momentum, higher timeframes for regime filter to avoid whipsaws
# - Target: 12-35 trades/year on 6h timeframe to stay within fee drag limits

name = "6h_1d_1w_ichimoku_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan = (pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
              pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun = (pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max() + 
             pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_b = ((pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                 pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2)
    
    # Current Kumo (cloud) boundaries - use Senkou spans shifted back 26 periods to align with price
    # The cloud plotted at time t is actually Senkou A/B from t-26
    senkou_a_shifted = senkou_a.shift(26)
    senkou_b_shifted = senkou_b.shift(26)
    
    # Upper and lower cloud boundaries
    upper_cloud = np.maximum(senkou_a_shifted, senkou_b_shifted)
    lower_cloud = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # Convert to numpy arrays
    tenkan = tenkan.values
    kijun = kijun.values
    upper_cloud = upper_cloud.values
    lower_cloud = lower_cloud.values
    
    # Calculate 1d and 1w close prices for regime filter
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    # Align HTF close prices to LTF
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Warmup period: need enough data for Ichimoku (52 + 26 shift)
    warmup = max(period_senkou_b, period_kijun) + 26
    
    for i in range(warmup, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or
            np.isnan(close_1d_aligned[i]) or np.isnan(close_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Ichimoku signals
        price_above_cloud = close[i] > upper_cloud[i]
        price_below_cloud = close[i] < lower_cloud[i]
        tk_bullish = tenkan[i] > kijun[i]
        tk_bearish = tenkan[i] < kijun[i]
        
        # Higher timeframe trend filter
        daily_bullish = close_1d_aligned[i] > close_1w_aligned[i]
        daily_bearish = close_1d_aligned[i] < close_1w_aligned[i]
        
        if position == 0:  # Flat - look for entry
            # Long: price above cloud, bullish TK cross, daily > weekly
            if price_above_cloud and tk_bullish and daily_bullish:
                position = 1
                signals[i] = 0.25
            # Short: price below cloud, bearish TK cross, daily < weekly
            elif price_below_cloud and tk_bearish and daily_bearish:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit: price re-enters cloud OR TK cross turns bearish
            if not price_above_cloud or not tk_bullish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit: price re-enters cloud OR TK cross turns bullish
            if not price_below_cloud or not tk_bearish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals