#!/usr/bin/env python3
# 6h_1d_donchian_breakout_v1
# Hypothesis: 6-hour Donchian breakout with 1-day trend filter and volume confirmation.
# Long when price breaks above 6h Donchian(20) high, with price above 1-day EMA(50) and volume > 1.5x average volume.
# Short when price breaks below 6h Donchian(20) low, with price below 1-day EMA(50) and volume > 1.5x average volume.
# Exit when price crosses the 6-day EMA(50) on 6h timeframe.
# Uses Donchian channels for breakout detection, 1-day EMA for trend filter, and volume surge for confirmation.
# Designed to generate ~20-40 trades/year to avoid excessive fees while capturing strong momentum moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_donchian_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for 20-period Donchian and 50-period EMA
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian channels (20-period)
    donchian_period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(donchian_period - 1, n):
        donchian_high[i] = np.max(high[i - donchian_period + 1:i + 1])
        donchian_low[i] = np.min(low[i - donchian_period + 1:i + 1])
    
    # Calculate 6h EMA(50) for exit
    ema_period = 50
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=ema_period, adjust=False, min_periods=ema_period).values
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1-day EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=ema_period, adjust=False, min_periods=ema_period).values
    
    # Align 1-day EMA(50) to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate average volume for volume confirmation (20-period)
    vol_ma_period = 20
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=vol_ma_period, min_periods=vol_ma_period).mean()
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        ema50 = ema_50[i]
        ema50_1d = ema_50_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 1:  # Long
            # Exit: price crosses below 6h EMA(50)
            if price < ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above 6h EMA(50)
            if price > ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation: current volume > 1.5x average volume
            vol_confirm = vol > 1.5 * vol_ma_val
            
            # Entry conditions
            # Bullish: price breaks above Donchian high, above 1-day EMA(50), with volume confirmation
            if price > donch_high and price > ema50_1d and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Bearish: price breaks below Donchian low, below 1-day EMA(50), with volume confirmation
            elif price < donch_low and price < ema50_1d and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals