#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily 100-period EMA for trend direction and 20-period ATR for volatility-based entries.
# Long when price > EMA100 and breaks above ATR-based upper band; short when price < EMA100 and breaks below lower band.
# Uses volume confirmation to filter false breakouts. Designed for moderate trade frequency (20-40/year) to balance edge and cost.
# Works in bull markets via trend following and in bear markets via short signals during downtrends.

name = "4h_EMA100_ATR_Bands_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA100 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA100 on daily close
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    # Align EMA100 to 4h timeframe
    ema_100_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Calculate ATR(20) on 4h data for volatility bands
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate upper and lower bands: EMA100 +/- 1.5 * ATR
    upper_band = ema_100_aligned + 1.5 * atr
    lower_band = ema_100_aligned - 1.5 * atr
    
    # Volume confirmation: 1.5x 20-period EMA of volume
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_threshold = vol_ema * 1.5
    vol_confirm = volume > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_100_aligned[i]) or 
            np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > EMA100 and breaks above upper band + volume confirmation
            if close[i] > ema_100_aligned[i] and close[i] > upper_band[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price < EMA100 and breaks below lower band + volume confirmation
            elif close[i] < ema_100_aligned[i] and close[i] < lower_band[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA100
            if close[i] < ema_100_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above EMA100
            if close[i] > ema_100_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals