#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Pivot Breakout with Weekly EMA Trend Filter and Volume Confirmation
# Uses Weekly pivot levels (S2/S1/PP/R1/R2) from 1w data for entry signals
# 1w EMA (50) provides trend direction filter to avoid counter-trend trades
# Volume confirmation (>1.8x average) ensures institutional participation
# Designed to work in both bull and bear markets by trading breakouts in direction of 1w trend
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for pivot levels and EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA (50) for trend direction
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate weekly pivot levels from prior week's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (high_1w + low_1w + close_1w) / 3.0
    # Support and resistance levels
    s1 = (2 * pp) - high_1w
    s2 = pp - (high_1w - low_1w)
    r1 = (2 * pp) - low_1w
    r2 = pp + (high_1w - low_1w)
    
    # Align pivot levels to daily timeframe (wait for weekly bar to close)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    
    # Volume confirmation: volume > 1.8x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for EMA and pivot calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade in direction of 1w EMA
        above_ema = price > ema_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume filter and above 1w EMA
            if price > r1_aligned[i] and vol > 1.8 * avg_vol[i] and above_ema:
                position = 1
                signals[i] = position_size
            # Short: price breaks below S1 with volume filter and below 1w EMA
            elif price < s1_aligned[i] and vol > 1.8 * avg_vol[i] and not above_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S1 (reversal) or below 1w EMA
            if price < s1_aligned[i] or price < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above R1 (reversal) or above 1w EMA
            if price > r1_aligned[i] or price > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_WeeklyPivot_Breakout_WeeklyEMA_Volume"
timeframe = "1d"
leverage = 1.0