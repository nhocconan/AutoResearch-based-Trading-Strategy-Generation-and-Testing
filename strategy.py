#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d 20-period Donchian breakout with weekly trend filter and volume confirmation.
# Donchian channels capture breakouts in both bull and bear markets.
# Weekly trend filter ensures we only trade in the direction of the higher timeframe trend.
# Volume confirmation ensures breakouts have conviction.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for multi-timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 1d data
    period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(period-1, n):
        donchian_high[i] = np.max(high[i-period+1:i+1])
        donchian_low[i] = np.min(low[i-period+1:i+1])
    
    # Calculate ATR (14-period) for volatility filter (optional, not used in entry)
    atr_period = 14
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Calculate weekly EMA (21-period) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = np.zeros(len(close_1w))
    for i in range(21, len(close_1w)):
        ema_1w[i] = np.mean(close_1w[i-20:i+1])  # Simple EMA approximation for speed
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(avg_volume[i]) or 
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        weekly_ema = ema_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price breaks above Donchian high + volume + price above weekly EMA
            if (price > donch_high and 
                volume_confirm and
                price > weekly_ema):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low + volume + price below weekly EMA
            elif (price < donch_low and 
                  volume_confirm and
                  price < weekly_ema):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low OR volume drops significantly
            if (price < donch_low or 
                vol < 0.5 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high OR volume drops significantly
            if (price > donch_high or 
                vol < 0.5 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_Breakout_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0