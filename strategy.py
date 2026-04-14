#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout with 1d EMA Trend Filter and Volume Confirmation
# Uses Camarilla pivot levels from daily data for entry signals
# 1d EMA (50) provides trend direction filter to avoid counter-trend trades
# Volume confirmation (>1.8x average volume) ensures institutional participation
# Designed to work in both bull and bear markets by trading breakouts in direction of 1d trend
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for pivot calculation and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA (50) for trend direction
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Camarilla formulas:
    # H4 = close + 1.1 * (high - low) / 2
    # H3 = close + 1.1 * (high - low) / 4
    # H2 = close + 1.1 * (high - low) / 6
    # H1 = close + 1.1 * (high - low) / 12
    # L1 = close - 1.1 * (high - low) / 12
    # L2 = close - 1.1 * (high - low) / 6
    # L3 = close - 1.1 * (high - low) / 4
    # L4 = close - 1.1 * (high - low) / 2
    
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Calculate pivot levels for previous day
    high_low_range = prev_high - prev_low
    
    H1 = prev_close + 1.1 * high_low_range / 12
    H2 = prev_close + 1.1 * high_low_range / 6
    H3 = prev_close + 1.1 * high_low_range / 4
    H4 = prev_close + 1.1 * high_low_range / 2
    
    L1 = prev_close - 1.1 * high_low_range / 12
    L2 = prev_close - 1.1 * high_low_range / 6
    L3 = prev_close - 1.1 * high_low_range / 4
    L4 = prev_close - 1.1 * high_low_range / 2
    
    # Align pivot levels to 12h timeframe (use previous day's levels)
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1)
    H2_aligned = align_htf_to_ltf(prices, df_1d, H2)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1)
    L2_aligned = align_htf_to_ltf(prices, df_1d, L2)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume confirmation: volume > 1.8x average volume (30-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(H1_aligned[i]) or np.isnan(L1_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade in direction of 1d EMA
        above_ema = price > ema_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above H3 with volume filter and above 1d EMA
            if price > H3_aligned[i] and vol > 1.8 * avg_vol[i] and above_ema:
                position = 1
                signals[i] = position_size
            # Short: price breaks below L3 with volume filter and below 1d EMA
            elif price < L3_aligned[i] and vol > 1.8 * avg_vol[i] and not above_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below H2 (reversal) or below 1d EMA
            if price < H2_aligned[i] or price < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above L2 (reversal) or above 1d EMA
            if price > L2_aligned[i] or price > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_1dEMA_Volume"
timeframe = "12h"
leverage = 1.0