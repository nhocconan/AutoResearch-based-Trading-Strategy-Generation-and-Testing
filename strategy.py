#!/usr/bin/env python3
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
    
    # Get 1h data for calculations (should already be 1h)
    # Get 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA34 for trend
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Get 1d data for weekly context
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for higher timeframe trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1h ATR for volatility
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1h volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_4h_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 4h and 1d
        trend_4h = close[i] > ema34_4h_aligned[i]
        trend_1d = close[i] > ema34_1d_aligned[i]
        
        # Only trade when both timeframes agree on trend
        trend_aligned = trend_4h and trend_1d
        
        # Long conditions: price above both EMAs + volume spike
        if trend_aligned and volume_spike[i]:
            if position <= 0:  # Not already long
                signals[i] = 0.20
                position = 1
            else:
                signals[i] = 0.20  # Maintain position
        # Short conditions: price below both EMAs + volume spike
        elif not trend_4h and not trend_1d and volume_spike[i]:
            if position >= 0:  # Not already short
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = -0.20  # Maintain position
        else:
            # Exit conditions: trend disagreement or no volume spike
            if position == 1 and (not trend_aligned or not volume_spike[i]):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (trend_4h or trend_1d or not volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals

name = "1h_EMA34_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0