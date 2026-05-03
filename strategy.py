#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 with volume > 1.8x 24-bar average and close > 1d EMA34 (uptrend)
# Short when price breaks below Camarilla S3 with volume > 1.8x 24-bar average and close < 1d EMA34 (downtrend)
# Exit when price returns to Camarilla Pivot level (mean reversion to equilibrium)
# Camarilla levels derived from 1d OHLC provide institutional support/resistance that work in both bull/bear markets via mean reversion and breakout patterns.
# Target: 50-150 total trades over 4 years = 12-37/year. Uses discrete sizing (0.25) to minimize fee churn.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from 1d OHLC (using previous day's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + (high-low)*1.1/2, R3 = close + (high-low)*1.1/4, R2 = close + (high-low)*1.1/6
    # R1 = close + (high-low)*1.1/12, PP = (high+low+close)/3
    # S1 = close - (high-low)*1.1/12, S2 = close - (high-low)*1.1/6, S3 = close - (high-low)*1.1/4
    # S4 = close - (high-low)*1.1/2
    rng = high_1d - low_1d
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r3 = close_1d + rng * 1.1 / 4.0
    camarilla_s3 = close_1d - rng * 1.1 / 4.0
    camarilla_pivot = camarilla_pp  # Use pivot as exit level
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volume confirmation (1.8x 24-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(34, 1) + 1  # EMA34(1d) + volume MA(24) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 with volume spike and close > 1d EMA34 (uptrend)
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_spike[i] and close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Camarilla S3 with volume spike and close < 1d EMA34 (downtrend)
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_spike[i] and close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns to Camarilla Pivot level (mean reversion)
            if close[i] <= camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns to Camarilla Pivot level (mean reversion)
            if close[i] >= camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals