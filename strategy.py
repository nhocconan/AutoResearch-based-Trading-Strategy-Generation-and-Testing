#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation
# Uses daily timeframe for signal generation with Donchian channel breakouts
# Weekly trend filter (price > weekly EMA50 for longs, < for shorts) ensures alignment with higher timeframe bias
# Volume confirmation (1.8x 20-period average) filters for institutional participation
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Weekly trend filter provides robustness in both bull and bear markets by avoiding counter-trend trades

name = "1d_Donchian20_1wEMA50_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channel (20-period)
    # Upper band: highest high over past 20 periods
    # Lower band: lowest low over past 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_band = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    lower_band = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Weekly trend filter: price > weekly EMA50 for longs, < for shorts
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above upper band + volume spike + price > weekly EMA50
            if close[i] > upper_band[i] and volume_spike[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band + volume spike + price < weekly EMA50
            elif close[i] < lower_band[i] and volume_spike[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below lower band or price < weekly EMA50
            if close[i] < lower_band[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above upper band or price > weekly EMA50
            if close[i] > upper_band[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals