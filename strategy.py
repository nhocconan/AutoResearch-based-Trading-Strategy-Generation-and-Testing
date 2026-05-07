#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator with 1-day trend filter and volume confirmation.
# The Alligator consists of three SMAs (Jaw=13, Teeth=8, Lips=5) representing market structure.
# In strong trends, the lines are well-separated and aligned (Jaw < Teeth < Lips for uptrend).
# In ranging markets, the lines intertwine and converge.
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1-day EMA34 (uptrend) AND volume > 1.5x 20-period average.
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1-day EMA34 (downtrend) AND volume > 1.5x 20-period average.
# Exit when the Alligator lines converge (|Lips - Jaw| < 0.1% of price) or volume filter fails.
# Designed for 12h timeframe to capture medium-term trends with low trade frequency (target: 12-37/year) to avoid fee drag.
# Uses 1-day EMA34 for trend filter to avoid counter-trend trades.
# Volume filter ensures participation and avoids low-conviction moves.
name = "12h_WilliamsAlligator_1dEMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: SMAs of median price (HL/2)
    median_price = (high + low) / 2
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values   # Lips: 5-period SMA
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values   # Teeth: 8-period SMA
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values   # Jaw: 13-period SMA
    
    # Alligator alignment: bullish when Lips > Teeth > Jaw, bearish when Lips < Teeth < Jaw
    bullish_alignment = (lips > teeth) & (teeth > jaw)
    bearish_alignment = (lips < teeth) & (teeth < jaw)
    
    # Convergence signal: when Alligator lines are close together (market ranging)
    # Avoid trading when |Lips - Jaw| < 0.1% of price
    convergence = np.abs(lips - jaw) < (0.001 * close)
    
    # 1-day EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(bullish_alignment[i]) or np.isnan(bearish_alignment[i]) or 
            np.isnan(convergence[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: bullish alignment, price above 1d EMA34, volume filter, not converging
            long_cond = bullish_alignment[i] and (close[i] > ema34_1d_aligned[i]) and volume_filter[i] and (not convergence[i])
            # Short conditions: bearish alignment, price below 1d EMA34, volume filter, not converging
            short_cond = bearish_alignment[i] and (close[i] < ema34_1d_aligned[i]) and volume_filter[i] and (not convergence[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish alignment OR convergence OR volume filter fails
            if bearish_alignment[i] or convergence[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish alignment OR convergence OR volume filter fails
            if bullish_alignment[i] or convergence[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals