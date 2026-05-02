#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d EMA50 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. In ranging markets, reversals from extreme
# levels (-80 for oversold, -20 for overbought) provide high-probability entries. The 1d EMA50
# filter ensures trades align with the higher-timeframe trend to avoid counter-trend whipsaws.
# Volume confirmation (1.5x 20-period average) filters low-momentum false signals.
# This combination works in both bull and bear markets by capturing mean reversion within the
# prevailing trend. Discrete sizing 0.25 targets ~50-100 trades over 4 years (12-25/year)
# to minimize fee drag while maintaining sufficient statistical significance.

name = "4h_WilliamsR_MeanReversion_1dEMA50_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams %R(14) on 4h
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Avoid division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate EMA(50) on 1d for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (1.5x 20-period average) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Williams %R and EMA calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R crosses above -80 (oversold reversal) with 1d uptrend
            long_entry = (williams_r[i] > -80) and (williams_r[i-1] <= -80)
            # Short entry: Williams %R crosses below -20 (overbought reversal) with 1d downtrend
            short_entry = (williams_r[i] < -20) and (williams_r[i-1] >= -20)
            
            # 1d EMA50 trend filter: close above/below EMA indicates trend direction
            ema_trend_up = close[i] > ema_50_1d_aligned[i]
            ema_trend_down = close[i] < ema_50_1d_aligned[i]
            
            if long_entry and ema_trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif short_entry and ema_trend_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (momentum loss) or trend reversal
            if (williams_r[i] < -50 and williams_r[i-1] >= -50) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (momentum loss) or trend reversal
            if (williams_r[i] > -50 and williams_r[i-1] <= -50) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals