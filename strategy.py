#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion strategy using 4h Bollinger Bands (20,2) for structure and 1d RSI(2) for extreme mean reversion timing.
# In 4h ranging markets (price within BB), wait for 1d RSI < 10 to go long (deep pullback) or > 90 to go short (extreme bounce).
# Volume confirmation filters weak moves. Session filter (08-20 UTC) reduces off-hours noise.
# Designed for low trade frequency (15-25/year) to minimize fee drag while capturing mean reversion in both bull and bear regimes.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Bollinger Bands (20,2) ===
    close_4h = df_4h['close'].values
    sma_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    bb_upper_aligned = align_htf_to_ltf(prices, df_4h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_4h, bb_lower)
    
    # === 1d Indicators: RSI(2) for extreme mean reversion ===
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_2 = 100 - (100 / (1 + rs))
    rsi_2_aligned = align_htf_to_ltf(prices, df_1d, rsi_2)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or
            np.isnan(rsi_2_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price within 4h Bollinger Bands (ranging market structure)
        # 2. 1d RSI(2) < 10 (extremely oversold)
        # 3. Volume confirmation
        if (close[i] >= bb_lower_aligned[i]) and (close[i] <= bb_upper_aligned[i]) and \
           (rsi_2_aligned[i] < 10) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price within 4h Bollinger Bands (ranging market structure)
        # 2. 1d RSI(2) > 90 (extremely overbought)
        # 3. Volume confirmation
        elif (close[i] >= bb_lower_aligned[i]) and (close[i] <= bb_upper_aligned[i]) and \
             (rsi_2_aligned[i] > 90) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_BB20_2_RSI2_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0