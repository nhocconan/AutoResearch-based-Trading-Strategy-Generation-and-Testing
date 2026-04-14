#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Bollinger Band squeeze breakout with 1-day volume confirmation
# Long when Bollinger Bands compress (low volatility) then break upward with volume
# Short when Bollinger Bands compress then break downward with volume
# Uses Bollinger Band width percentile to detect squeeze, breakout confirmed by volume spike
# Works in both bull and bear markets by capturing volatility breakouts after consolidation
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 6h and 1d data ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Bollinger Bands on 6h: 20-period, 2 std dev
    close_6h = df_6h['close'].values
    sma_20 = pd.Series(close_6h).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close_6h).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    bb_width = upper_bb - lower_bb
    
    # Calculate Bollinger Band width percentile (50-period lookback) to detect squeeze
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Calculate 1-day volume average (20-period) for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_6h, bb_width_percentile)
    upper_bb_aligned = align_htf_to_ltf(prices, df_6h, upper_bb.values)
    lower_bb_aligned = align_htf_to_ltf(prices, df_6h, lower_bb.values)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Bollinger squeeze condition: BB width in lowest 20% percentile (compression)
        squeeze = bb_width_percentile_aligned[i] < 20
        
        # Volume confirmation: current volume > 1.5x 20-day average volume
        vol_confirm = vol > (1.5 * vol_ma_20_aligned[i])
        
        if position == 0:
            # Long entry: squeeze breakout upward with volume
            if squeeze and (price > upper_bb_aligned[i]) and vol_confirm:
                position = 1
                signals[i] = position_size
            # Short entry: squeeze breakout downward with volume
            elif squeeze and (price < lower_bb_aligned[i]) and vol_confirm:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle of Bollinger Bands (mean reversion)
            bb_middle = (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2
            if price < bb_middle:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle of Bollinger Bands
            bb_middle = (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2
            if price > bb_middle:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_BollingerSqueeze_VolumeBreakout"
timeframe = "6h"
leverage = 1.0