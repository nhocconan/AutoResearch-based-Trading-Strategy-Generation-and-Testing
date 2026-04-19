#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Keltner Channel breakout with 1d volume confirmation and 1d trend filter
# - Keltner Channel: EMA(20) ± 2*ATR(10) for volatility-based breakout
# - Long when price breaks above upper band in uptrend (price > 1d EMA50)
# - Short when price breaks below lower band in downtrend (price < 1d EMA50)
# - Volume confirmation: current 4h volume > 1.5x 20-period average 1d volume (scaled)
# - Exit when price crosses back through EMA(20) middle band
# - Designed to capture volatility expansion moves in both bull and bear markets
# - Target: 20-35 trades/year to stay within fee limits

name = "4h_KeltnerBreakout_1dVolume_1dTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and trend filters
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Keltner Channel components (using 4h data)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_10 = pd.Series(np.maximum.reduce([
        high[1:] - low[:-1],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[:-1] - close[:-1])
    ])).ewm(span=10, adjust=False, min_periods=10).mean()
    # Prepend first ATR value to match length
    atr_10 = np.concatenate([[np.nan], atr_10.values]) if len(atr_10) > 0 else np.array([np.nan])
    
    upper_keltner = ema_20 + 2 * atr_10
    lower_keltner = ema_20 - 2 * atr_10
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(ema_20[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 4h: 1d has 6x 4h bars, so divide by 6
        volume_factor = vol_ma_1d_aligned[i] / 6.0 if vol_ma_1d_aligned[i] > 0 else 0
        volume_filter = volume[i] > 1.5 * volume_factor
        
        if position == 0:
            # Look for long entry: uptrend + price breaks above upper Keltner + volume
            if close[i] > ema_50_1d_aligned[i] and close[i] > upper_keltner[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend + price breaks below lower Keltner + volume
            elif close[i] < ema_50_1d_aligned[i] and close[i] < lower_keltner[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price crosses back below EMA(20) middle band
            if close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price crosses back above EMA(20) middle band
            if close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals