#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h price channel breakout with volume confirmation and daily trend filter.
# Uses weekly ATR filter to avoid low-volatility false breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
# Designed to work in both bull and bear markets by combining breakout logic with trend filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for volatility filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Price channel (Donchian-like) on 12h: 20-period high/low
    period = 20
    price_high = np.full(n, np.nan)
    price_low = np.full(n, np.nan)
    
    for i in range(period-1, n):
        price_high[i] = np.max(high[i-period+1:i+1])
        price_low[i] = np.min(low[i-period+1:i+1])
    
    # Weekly ATR (14-period) for volatility filter
    atr_period = 14
    tr = np.zeros(len(df_1w))
    for i in range(1, len(df_1w)):
        tr[i] = max(
            df_1w['high'].iloc[i] - df_1w['low'].iloc[i],
            abs(df_1w['high'].iloc[i] - df_1w['close'].iloc[i-1]),
            abs(df_1w['low'].iloc[i] - df_1w['close'].iloc[i-1])
        )
    
    atr_1w = np.zeros(len(df_1w))
    for i in range(atr_period, len(df_1w)):
        atr_1w[i] = np.mean(tr[i-atr_period+1:i+1])
    
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Daily SMA (20-period) for trend filter
    close_1d = df_1d['close'].values
    sma_1d = np.zeros(len(close_1d))
    for i in range(20, len(close_1d)):
        sma_1d[i] = np.mean(close_1d[i-20:i])
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Average volume (20-period) for confirmation
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(price_high[i]) or np.isnan(price_low[i]) or 
            np.isnan(atr_1w_aligned[i]) or np.isnan(sma_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ph = price_high[i]
        pl = price_low[i]
        atr_val = atr_1w_aligned[i]
        daily_sma = sma_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        # Volatility filter: ATR > 0.5 * average ATR (avoid low-vol false breakouts)
        # Calculate average ATR for dynamic threshold
        if i >= 50:
            atr_avg = np.mean(atr_1w_aligned[max(0, i-50):i])
            vol_filter = atr_val > 0.5 * atr_avg
        else:
            vol_filter = atr_val > 0  # fallback for early bars
        
        if position == 0:
            # Long: price breaks above channel + volume + price above daily SMA + sufficient volatility
            if (price > ph and 
                volume_confirm and 
                vol_filter and
                price > daily_sma):
                position = 1
                signals[i] = position_size
            # Short: price breaks below channel + volume + price below daily SMA + sufficient volatility
            elif (price < pl and 
                  volume_confirm and 
                  vol_filter and
                  price < daily_sma):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below channel OR volume drops significantly
            if (price < pl or 
                vol < 0.4 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above channel OR volume drops significantly
            if (price > ph or 
                vol < 0.4 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_Price_Channel_Volume_Volatility_Filter_v1"
timeframe = "12h"
leverage = 1.0