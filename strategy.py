#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Bollinger Band squeeze breakout with 4h trend filter and volume confirmation
# Long when price breaks above upper BB (20,2) + 4h EMA(50) uptrend + volume spike
# Short when price breaks below lower BB (20,2) + 4h EMA(50) downtrend + volume spike
# Bollinger squeeze identifies low volatility periods primed for breakout
# 4h trend filter ensures alignment with higher timeframe momentum
# Volume spike confirms institutional participation
# Session filter (08-20 UTC) reduces noise trades
# Targets 60-150 total trades over 4 years (15-37/year) to avoid fee drag

name = "1h_BB_Squeeze_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data once for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Bollinger Bands (20,2) on 1h
    ma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma20 + (2 * std20)
    lower_bb = ma20 - (2 * std20)
    
    # Squeeze condition: BB width < 50-period average width
    bb_width = upper_bb - lower_bb
    width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze = bb_width < width_ma
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ma20[i]) or 
            np.isnan(std20[i]) or np.isnan(vol_ma[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_4h_val = ema50_4h_aligned[i]
        upper = upper_bb[i]
        lower = lower_bb[i]
        sqz = squeeze[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: squeeze + price > upper BB + 4h uptrend + volume spike
            if sqz and close[i] > upper and close[i] > ema50_4h_val and vol_spike:
                signals[i] = 0.20
                position = 1
            # Enter short: squeeze + price < lower BB + 4h downtrend + volume spike
            elif sqz and close[i] < lower and close[i] < ema50_4h_val and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price < middle BB OR 4h trend turns down
            if close[i] < ma20[i] or close[i] < ema50_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price > middle BB OR 4h trend turns up
            if close[i] > ma20[i] or close[i] > ema50_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals