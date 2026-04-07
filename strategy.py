#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Williams %R + 1d EMA Trend + Volume Spike
# Hypothesis: Williams %R identifies overbought/oversold conditions on 6h.
# In trending markets (1d EMA slope), we take pullbacks in trend direction.
# In ranging markets (flat 1d EMA), we fade extremes.
# Volume spike confirms conviction. Works in bull/bear by following trend.
# Target: 15-30 trades/year (60-120 total over 4 years).

name = "6h_williams_r_1d_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d EMA slope for trend vs range detection
    ema_slope = np.zeros_like(ema_50_1d_aligned)
    ema_slope[1:] = (ema_50_1d_aligned[1:] - ema_50_1d_aligned[:-1]) / ema_50_1d_aligned[:-1]
    
    # Williams %R on 6h (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range=0
    )
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(ema_slope[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: Williams %R exits overbought OR trend turns bearish
            if williams_r[i] > -20 or ema_slope[i] < -0.001:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: Williams %R exits oversold OR trend turns bullish
            if williams_r[i] < -80 or ema_slope[i] > 0.001:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Determine market regime: trending vs ranging
                is_trending = abs(ema_slope[i]) > 0.0005  # significant slope
                
                if is_trending:
                    # Trending market: pullback entry in trend direction
                    # Long: pullback to oversold in uptrend
                    if williams_r[i] < -80 and ema_slope[i] > 0.001:
                        position = 1
                        signals[i] = 0.25
                    # Short: pullback to overbought in downtrend
                    elif williams_r[i] > -20 and ema_slope[i] < -0.001:
                        position = -1
                        signals[i] = -0.25
                else:
                    # Ranging market: fade extremes
                    # Long at oversold
                    if williams_r[i] < -85:
                        position = 1
                        signals[i] = 0.25
                    # Short at overbought
                    elif williams_r[i] > -15:
                        position = -1
                        signals[i] = -0.25
    
    return signals