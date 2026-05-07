#!/usr/bin/env python3
name = "1d_WideRange_Fade_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA10 trend filter
    ema_10_1w = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Daily range expansion filter: today's range > 1.5 * 20-day average range
    daily_range = high - low
    avg_range = pd.Series(daily_range).rolling(window=20, min_periods=20).mean().values
    wide_range = daily_range > (avg_range * 1.5)
    
    # Mean reversion: fade extreme closes relative to 5-day VWAP
    typical_price = (high + low + close) / 3
    vwap_num = (typical_price * volume).cumsum()
    vwap_den = volume.cumsum()
    vwap = vwap_num / vwap_den
    # Reset VWAP every 5 days
    vwap_reset = pd.Series(vwap).groupby(np.arange(len(vwap)) // 5).transform(lambda x: x.ffill())
    price_vwap_ratio = close / vwap_reset
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for range average
    
    for i in range(start_idx, n):
        if np.isnan(ema_10_1w_aligned[i]) or np.isnan(avg_range[i]) or np.isnan(price_vwap_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Fade extreme closes on wide range days
            # Short when price closes significantly above VWAP on wide range day
            if price_vwap_ratio[i] > 1.02 and wide_range[i] and close[i] < ema_10_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            # Long when price closes significantly below VWAP on wide range day
            elif price_vwap_ratio[i] < 0.98 and wide_range[i] and close[i] > ema_10_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
        elif position != 0:
            # Exit when price returns to VWAP or weekly trend reverses
            if position == 1:
                if price_vwap_ratio[i] >= 0.995 or close[i] <= ema_10_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_vwap_ratio[i] >= 1.005 or close[i] >= ema_10_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals