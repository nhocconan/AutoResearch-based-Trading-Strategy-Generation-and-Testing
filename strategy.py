#!/usr/bin/env python3
"""
4h_Bollinger_Bands_Squeeze_Reversion
Hypothesis: Bollinger Band squeeze (low volatility) followed by breakout with volume capture
mean reversion moves in ranging markets, while avoiding false breakouts in trends.
Works in both bull and bear markets by combining volatility contraction (BB width < 20th percentile)
with mean reversion at bands (price touching upper/lower BB) and volume confirmation.
Uses 1-day trend filter to avoid counter-trend trades in strong trends.
Target: 20-40 trades/year to minimize fee drag.
"""

name = "4h_Bollinger_Bands_Squeeze_Reversion"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for 1-day trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean()
    std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std()
    upper_band = sma + (bb_std * std_dev)
    lower_band = sma - (bb_std * std_dev)
    bb_width = upper_band - lower_band
    
    # Bollinger Band squeeze: width < 20th percentile of last 50 periods
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).quantile(0.20)
    squeeze = bb_width < bb_width_percentile.values
    
    # Mean reversion signals: price touches upper/lower band
    touches_upper = close >= upper_band.values
    touches_lower = close <= lower_band.values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Squeeze + touches lower band + volume + below 1-day EMA50 (mean reversion in downtrend)
            if (squeeze[i] and 
                touches_lower[i] and 
                volume_confirmed[i] and 
                close[i] < trend_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Squeeze + touches upper band + volume + above 1-day EMA50 (mean reversion in uptrend)
            elif (squeeze[i] and 
                  touches_upper[i] and 
                  volume_confirmed[i] and 
                  close[i] > trend_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to middle (SMA) or squeeze ends
            if (close[i] >= sma.values[i] or 
                not squeeze[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to middle (SMA) or squeeze ends
            if (close[i] <= sma.values[i] or 
                not squeeze[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals