#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with 4h volume confirmation and 4h ATR volatility filter.
# Uses daily Donchian channels (20) for breakout direction, 4h volume spike for confirmation,
# and 4h ATR to avoid low volatility periods. Designed for 20-40 trades/year to minimize fee drag
# and work in both bull and bear markets by trading breakouts with volatility filtering.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels (20) on daily data
    upper_20 = np.full(len(high_1d), np.nan)
    lower_20 = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        upper_20[i] = np.max(high_1d[i-20:i])
        lower_20[i] = np.min(low_1d[i-20:i])
    
    # Align daily Donchian to 4h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Calculate 4h ATR for volatility filter
    tr_4h_1 = high - low
    tr_4h_2 = np.abs(high - np.roll(close, 1))
    tr_4h_3 = np.abs(low - np.roll(close, 1))
    tr_4h_1[0] = high[0] - low[0]
    tr_4h_2[0] = np.abs(high[0] - close[0])
    tr_4h_3[0] = np.abs(low[0] - close[0])
    tr_4h = np.maximum(tr_4h_1, np.maximum(tr_4h_2, tr_4h_3))
    atr_4h = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # need daily Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(atr_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0 * 20-period average (spike)
        vol_spike = volume[i] > 2.0 * vol_ma[i]
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_4h[i] > 0.5 * np.nanmedian(atr_4h[max(0, i-100):i+1])
        
        if position == 0:
            # Long entry: price breaks above daily upper Donchian with volume spike and vol filter
            if (close[i] > upper_20_aligned[i] and 
                vol_spike and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below daily lower Donchian with volume spike and vol filter
            elif (close[i] < lower_20_aligned[i] and 
                  vol_spike and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below daily lower Donchian or volatility drops
            if close[i] < lower_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above daily upper Donchian or volatility drops
            if close[i] > upper_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DailyDonchian20_VolumeSpike_VolatilityFilter"
timeframe = "4h"
leverage = 1.0