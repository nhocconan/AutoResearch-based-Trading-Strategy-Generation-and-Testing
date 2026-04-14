#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d EMA200 trend filter and 4h Donchian breakout with volume confirmation.
# EMA200 from daily timeframe filters trades to align with long-term trend (avoid counter-trend trades).
# Donchian breakout from 4h provides entry signals with high probability of continuation.
# Volume confirmation (>1.5x 20-period average) reduces false breakouts.
# ATR-based stop loss manages risk via signal=0 when price moves against position.
# Designed to work in both bull and bear markets by using daily EMA200 filter to avoid counter-trend trades.
# Target: 20-30 trades/year per symbol (80-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate EMA200 on 1d close
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 4h timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss (10-period)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(200, 20)  # Need EMA200 and Donchian
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_200_aligned[i]) or 
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for breakouts above 4h Donchian high or below 4h Donchian low
            # Only trade in direction of daily EMA200 (trend filter)
            
            # Long: price breaks above 4h Donchian high AND price above EMA200
            if (close[i] > donchian_high[i] and 
                close[i] > ema_200_aligned[i] and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below 4h Donchian low AND price below EMA200
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_200_aligned[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to 4h Donchian low or closes below EMA200
            if (close[i] <= donchian_low[i] or 
                close[i] < ema_200_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to 4h Donchian high or closes above EMA200
            if (close[i] >= donchian_high[i] or 
                close[i] > ema_200_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dEMA200_4hDonchian_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0