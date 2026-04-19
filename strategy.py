#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout (20) with 1d EMA50 trend filter + volume confirmation + ATR stop.
# Long when price breaks above Donchian(20) high, EMA50 rising, volume > 2x average.
# Short when price breaks below Donchian(20) low, EMA50 falling, volume > 2x average.
# Exit on opposite breakout or volatility drop.
# Uses 1d EMA50 for trend filter to avoid counter-trend trades in both bull/bear markets.
# Target: ~25-35 trades/year to stay under 400 total over 4 years.
name = "4h_Donchian_EMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for stop (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    # Get 1d EMA50 (trend filter) - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # EMA50 slope (rising/falling)
    ema_slope = np.diff(ema_50_1d_aligned, prepend=ema_50_1d_aligned[0])
    ema_rising = ema_slope > 0
    ema_falling = ema_slope < 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_ma[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high + EMA50 rising + volume spike
            if (close[i] > donchian_high[i] and 
                ema_rising[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low + EMA50 falling + volume spike
            elif (close[i] < donchian_low[i] and 
                  ema_falling[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit on breakout below Donchian low OR EMA50 falling
            if (close[i] < donchian_low[i]) or (not ema_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit on breakout above Donchian high OR EMA50 rising
            if (close[i] > donchian_high[i]) or (not ema_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals