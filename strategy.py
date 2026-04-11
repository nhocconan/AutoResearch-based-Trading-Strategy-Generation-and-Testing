# 12h_1w_1d_camarilla_breakout_volume_v1
# Strategy: 12h Camarilla pivot breakout with volume confirmation and 1w/1d trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels provide high-probability support/resistance levels.
# Breakouts above/below key levels (H3/L3) with volume confirmation and trend alignment
# yield reliable moves. Uses 1w trend filter to avoid counter-trend trades and 1d
# regime filter to ensure momentum alignment. Low frequency (~15-30/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w and 1d data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d EMA(200) for regime filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Camarilla pivot levels from 1d OHLC
    # Camarilla levels: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    # Using typical multiplier 1.1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate pivot range
    rng = high_1d - low_1d
    camarilla_multiplier = 1.1
    
    # H3 and L3 levels (key breakout levels)
    h3 = close_1d_vals + camarilla_multiplier * rng * 1.1 / 4
    l3 = close_1d_vals - camarilla_multiplier * rng * 1.1 / 4
    
    # Align H3/L3 to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Regime filter: price above/below 1d EMA200
        bull_regime = close[i] > ema_200_1d_aligned[i]
        bear_regime = close[i] < ema_200_1d_aligned[i]
        
        # Entry logic: Camarilla breakout + volume + trend/regime alignment
        if (close[i] > h3_aligned[i] and  # Break above H3 resistance
            vol_confirm[i] and uptrend and bull_regime and position != 1):
            position = 1
            signals[i] = 0.25
        elif (close[i] < l3_aligned[i] and  # Break below L3 support
              vol_confirm[i] and downtrend and bear_regime and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: trend or regime reversal
        elif position == 1 and (not uptrend or not bull_regime):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not downtrend or not bear_regime):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals