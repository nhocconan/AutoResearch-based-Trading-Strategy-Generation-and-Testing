#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud strategy with 1d/1w filter
# Long when: Tenkan > Kijun, price above Kumo (cloud), and Kumo is bullish (Senkou A > Senkou B)
# Short when: Tenkan < Kijun, price below Kumo, and Kumo is bearish (Senkou A < Senkou B)
# Uses 1d EMA(50) for higher timeframe trend alignment
# Ichimoku parameters: tenkan=9, kijun=26, senkou=52 (standard)
# Target: 50-150 total trades over 4 years with clear trend following in both bull/bear markets

name = "6h_ichimoku_1d_ema_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Ichimoku Cloud components (standard periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(52, n):  # Need 52 periods for Senkou B
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Ichimoku conditions
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        price_above_kumo = close[i] > max(senkou_a[i], senkou_b[i])
        price_below_kumo = close[i] < min(senkou_a[i], senkou_b[i])
        kumo_bullish = senkou_a[i] > senkou_b[i]
        kumo_bearish = senkou_a[i] < senkou_b[i]
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Tenkan crosses below Kijun or price below Kumo
            elif tenkan_below_kijun or price_below_kumo:
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
            # Exit: Tenkan crosses above Kijun or price above Kumo
            elif tenkan_above_kijun or price_above_kumo:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: need alignment with 1d EMA trend
            # Long: Tenkan > Kijun, price above Kumo, Kumo bullish, 1d EMA rising
            if (tenkan_above_kijun and price_above_kumo and kumo_bullish and 
                ema_1d_aligned[i] > ema_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Tenkan < Kijun, price below Kumo, Kumo bearish, 1d EMA falling
            elif (tenkan_below_kijun and price_below_kumo and kumo_bearish and 
                  ema_1d_aligned[i] < ema_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals