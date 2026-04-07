#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Ichimoku Cloud with Weekly Trend Filter
# Hypothesis: Ichimoku (Tenkan/Kijun cross + Cloud) provides high-probability entries
# when aligned with weekly trend. Works in bull/bear by following higher-timeframe trend.
# Target: 15-35 trades/year (60-140 total over 4 years).

name = "6h_ichimoku_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    highest_tenkan = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_tenkan = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    highest_kijun = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_kijun = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    highest_senkou_b = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_senkou_b = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = (highest_senkou_b + lowest_senkou_b) / 2
    
    # Weekly EMA(50) for trend filter
    ema_50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False).mean().values
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(tenkan_period, kijun_period, senkou_span_b_period), n):
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_span_a[i]) or
            np.isnan(senkou_span_b[i]) or np.isnan(ema_50_weekly_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: Tenkan crosses below Kijun OR price falls below Cloud
            if tenkan[i] < kijun[i] or close[i] < senkou_span_a[i] or close[i] < senkou_span_b[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: Tenkan crosses above Kijun OR price rises above Cloud
            if tenkan[i] > kijun[i] or close[i] > senkou_span_a[i] or close[i] > senkou_span_b[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Bullish: Tenkan crosses above Kijun AND price above Cloud AND weekly uptrend
                if (tenkan[i] > kijun[i] and 
                    close[i] > senkou_span_a[i] and close[i] > senkou_span_b[i] and
                    close[i] > ema_50_weekly_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Bearish: Tenkan crosses below Kijun AND price below Cloud AND weekly downtrend
                elif (tenkan[i] < kijun[i] and 
                      close[i] < senkou_span_a[i] and close[i] < senkou_span_b[i] and
                      close[i] < ema_50_weekly_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals