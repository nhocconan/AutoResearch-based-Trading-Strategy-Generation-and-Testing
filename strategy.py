#!/usr/bin/env python3
"""
1h_4h_1d_PriceAction_Volume_Trend_v1
Hypothesis: On 1h timeframe, use 4h price action (higher highs/lows) and 1d trend (EMA50) for directional bias, 
with volume confirmation for entry timing. Long when: price makes higher high on 4h, above 1d EMA50, 
and volume spike. Short when: price makes lower low on 4h, below 1d EMA50, and volume spike. 
Exit on opposite signal or volume dry-up. Works in bull/bear by following 4h structure and 1d trend.
Designed to avoid overtrading: uses higher timeframe for direction, 1h only for timing.
Target: 15-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data for structure (higher highs/lows)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h higher highs and lower lows
    hh_4h = np.full_like(df_4h['close'], np.nan)  # Higher high
    ll_4h = np.full_like(df_4h['close'], np.nan)  # Lower low
    
    for i in range(2, len(df_4h)):
        if (np.isnan(df_4h['high'].iloc[i]) or np.isnan(df_4h['low'].iloc[i]) or
            np.isnan(df_4h['high'].iloc[i-1]) or np.isnan(df_4h['low'].iloc[i-1]) or
            np.isnan(df_4h['high'].iloc[i-2]) or np.isnan(df_4h['low'].iloc[i-2])):
            continue
        # Higher high: current high > previous high AND previous high > high two bars ago
        if (df_4h['high'].iloc[i] > df_4h['high'].iloc[i-1] and 
            df_4h['high'].iloc[i-1] > df_4h['high'].iloc[i-2]):
            hh_4h.iloc[i] = df_4h['high'].iloc[i]
        # Lower low: current low < previous low AND previous low < low two bars ago
        if (df_4h['low'].iloc[i] < df_4h['low'].iloc[i-1] and 
            df_4h['low'].iloc[i-1] < df_4h['low'].iloc[i-2]):
            ll_4h.iloc[i] = df_4h['low'].iloc[i]
    
    # Load 1d data for trend (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close']
    ema_50 = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 1h timeframe
    hh_4h_aligned = align_htf_to_ltf(prices, df_4h, hh_4h.values)
    ll_4h_aligned = align_htf_to_ltf(prices, df_4h, ll_4h.values)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume spike detection (20-period average)
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    vol_ratio = np.where(vol_ma_20 > 0, volume / vol_ma_20, 0)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    for i in range(50, n):
        # Skip if not in trading session or missing data
        if not in_session[i]:
            signals[i] = 0.0
            continue
        if (np.isnan(hh_4h_aligned[i]) or np.isnan(ll_4h_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: 4h higher high, above 1d EMA50, volume spike
            if (not np.isnan(hh_4h_aligned[i]) and 
                close[i] > ema_50_aligned[i] and 
                vol_ratio[i] > 2.0):
                position = 1
                signals[i] = position_size
            # Short: 4h lower low, below 1d EMA50, volume spike
            elif (not np.isnan(ll_4h_aligned[i]) and 
                  close[i] < ema_50_aligned[i] and 
                  vol_ratio[i] > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: opposite signal (lower low) or volume dry-up
            if (not np.isnan(ll_4h_aligned[i]) and 
                close[i] < ema_50_aligned[i]) or vol_ratio[i] < 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: opposite signal (higher high) or volume dry-up
            if (not np.isnan(hh_4h_aligned[i]) and 
                close[i] > ema_50_aligned[i]) or vol_ratio[i] < 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_PriceAction_Volume_Trend_v1"
timeframe = "1h"
leverage = 1.0