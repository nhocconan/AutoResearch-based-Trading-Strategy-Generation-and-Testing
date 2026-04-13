#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + 1w trend filter (EMA200)
# Long: price breaks above Donchian(20) high + volume > 2x avg volume + price > 1w EMA200
# Short: price breaks below Donchian(20) low + volume > 2x avg volume + price < 1w EMA200
# Uses 1w EMA200 as long-term trend filter to avoid counter-trend trades.
# Volume spike confirms breakout strength.
# Target: 20-50 trades/year for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) on 4h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # 1d volume spike filter
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):  # 20-period = ~5 days
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # 1w EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema_trend = ema_200_1w_aligned[i]
        
        # Volume confirmation: current volume > 2x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: break above upper band + volume confirmation + above weekly EMA200
            if (price > upper and 
                volume_confirm and
                price > ema_trend):
                position = 1
                signals[i] = position_size
            # Short: break below lower band + volume confirmation + below weekly EMA200
            elif (price < lower and 
                  volume_confirm and
                  price < ema_trend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below midpoint of Donchian channel
            midpoint = (upper + lower) / 2
            if price < midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises above midpoint of Donchian channel
            midpoint = (upper + lower) / 2
            if price > midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1w_Donchian_Volume_Trend"
timeframe = "4h"
leverage = 1.0