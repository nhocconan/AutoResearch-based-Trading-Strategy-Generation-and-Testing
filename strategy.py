#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Bollinger Band breakout with 4h trend filter and volume confirmation
# Uses 4h trend direction (price above/below 4h EMA50) to filter breakouts.
# Entry: price breaks above upper BB(20,2) with volume spike in uptrend (long) or breaks below lower BB in downtrend (short).
# Exit: reverse signal or stop via signal=0. Target: 15-30 trades/year per symbol.
# Works in bull/bear: follows 4h trend, avoids counter-trend breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Bollinger Bands (20, 2) on 1h close
    close_series = pd.Series(close)
    sma20 = close_series.rolling(window=20, min_periods=20).mean()
    std20 = close_series.rolling(window=20, min_periods=20).std()
    upper_bb = (sma20 + 2 * std20).values
    lower_bb = (sma20 - 2 * std20).values
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = close_series.rolling(window=20, min_periods=20).mean()  # using close for volume MA approximation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above upper BB in uptrend (close > 4h EMA50) + volume
        if (close[i] > upper_bb[i] and 
            close[i] > ema50_4h_aligned[i] and   # Uptrend filter
            volume_filter[i]):
            signals[i] = 0.20
            position = 1
        # Short conditions: price breaks below lower BB in downtrend (close < 4h EMA50) + volume
        elif (close[i] < lower_bb[i] and 
              close[i] < ema50_4h_aligned[i] and   # Downtrend filter
              volume_filter[i]):
            signals[i] = -0.20
            position = -1
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals

name = "1h_BollingerBreakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0