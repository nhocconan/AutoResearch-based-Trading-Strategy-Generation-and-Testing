#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w Trend Filter + Volume Spike
# Long when price breaks above Donchian upper band, 1w uptrend, volume > 1.5x average
# Short when price breaks below Donchian lower band, 1w downtrend, volume > 1.5x average
# Donchian channels capture breakouts; volume confirms strength; 1w trend filter ensures alignment with higher timeframe momentum
# Targets 30-100 total trades over 4 years (7-25/year) to minimize fee drag and improve generalization

name = "1d_Donchian20_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    weekly_close = df_1w['close'].values
    ema50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Donchian(20) channels
    upper_band = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_val = upper_band[i]
        lower_val = lower_band[i]
        ema50_1w_val = ema50_1w_aligned[i]
        vol_spike_val = vol_spike[i]
        close_val = close[i]
        
        if position == 0:
            # Enter long: breakout above upper band, 1w uptrend, volume spike
            if close_val > upper_val and ema50_1w_val > 0 and vol_spike_val:
                signals[i] = 0.25
                position = 1
            # Enter short: breakdown below lower band, 1w downtrend, volume spike
            elif close_val < lower_val and ema50_1w_val < 0 and vol_spike_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: breakdown below lower band or 1w trend down
            if close_val < lower_val or ema50_1w_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: breakout above upper band or 1w trend up
            if close_val > upper_val or ema50_1w_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals