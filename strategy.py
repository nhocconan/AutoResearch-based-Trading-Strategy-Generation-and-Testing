# #!/usr/bin/env python3
"""
6h Bollinger Band Squeeze Breakout with Volume Spike and Daily Trend Filter
Hypothesis: Bollinger Band squeeze (low volatility) followed by breakout with volume
captures explosive moves. Daily EMA50 filters direction to avoid counter-trend trades.
Works in bull markets (breakouts continue) and bear markets (breakdowns continue).
Designed for 12-37 trades/year on 6h timeframe with strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA filter (once before loop)
    df_d = get_htf_data(prices, '1d')
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_stddev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + (bb_std * bb_stddev)
    lower = sma - (bb_std * bb_stddev)
    bb_width = (upper - lower) / sma  # normalized width
    
    # Bollinger Squeeze: width below 20-period average of width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Daily EMA50 for trend filter
    ema_50 = pd.Series(df_d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_d, ema_50)
    
    # Volume spike: 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(sma[i]) or 
            np.isnan(upper[i]) or
            np.isnan(lower[i]) or
            np.isnan(ema_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema = ema_aligned[i]
        
        if position == 0:
            # Long: break above upper BB with volume spike and squeeze + price above EMA50
            if price > upper[i] and volume_spike[i] and squeeze[i] and price > ema:
                signals[i] = 0.25
                position = 1
            # Short: break below lower BB with volume spike and squeeze + price below EMA50
            elif price < lower[i] and volume_spike[i] and squeeze[i] and price < ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to middle Bollinger Band (mean reversion)
            if price < sma[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to middle Bollinger Band
            if price > sma[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_BollingerSqueeze_Breakout_Volume_EMA50"
timeframe = "6h"
leverage = 1.0