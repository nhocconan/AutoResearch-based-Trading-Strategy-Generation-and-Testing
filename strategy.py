#!/usr/bin/env python3
# 12h Bollinger Band Squeeze Breakout with Volume Confirmation and 1d Trend Filter
# Hypothesis: Bollinger Band squeezes on 12h chart precede explosive moves.
# Breakouts above upper band with volume > 1.5x 20-period average and in 1d uptrend (close > EMA50) capture bullish momentum.
# Breakdowns below lower band with volume > 1.5x 20-period average and in 1d downtrend (close < EMA50) capture bearish momentum.
# Works in bull/bear markets by trading breakouts from low volatility regimes with trend alignment to avoid false signals.
# Target: 15-30 trades/year per symbol.

name = "12h_bb_squeeze_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_d = get_htf_data(prices, '1d')
    close_d = df_d['close'].values
    
    # Calculate 12h Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + (bb_std * std)
    lower = sma - (bb_std * std)
    
    # Calculate Bollinger Band Width for squeeze detection
    bb_width = (upper - lower) / sma
    bb_width = np.where(sma == 0, 0, bb_width)
    
    # Calculate 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA50 for trend filter
    ema50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 50
    
    for i in range(start_idx, n):
        # Get aligned daily EMA50 for current 12h bar
        ema50 = align_htf_to_ltf(prices, df_d, ema50_d)[i]
        
        # Skip if any required data is NaN
        if np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(vol_ma[i]) or np.isnan(ema50):
            signals[i] = 0.0
            continue
        
        # Volume breakout condition: current volume > 1.5x 20-period average
        vol_breakout = volume[i] > 1.5 * vol_ma[i]
        
        # Bollinger Band squeeze condition: BB width < 0.05 (5%)
        squeeze = bb_width[i] < 0.05
        
        if position == 1:  # Long position
            # Exit if price closes below middle Bollinger Band (mean reversion)
            if close[i] < sma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price closes above middle Bollinger Band (mean reversion)
            if close[i] > sma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long above upper band with volume confirmation and 1d uptrend
            if close[i] > upper[i] and vol_breakout and squeeze and close_d[i] > ema50[i]:
                position = 1
                signals[i] = 0.25
            # Breakdown short below lower band with volume confirmation and 1d downtrend
            elif close[i] < lower[i] and vol_breakout and squeeze and close_d[i] < ema50[i]:
                position = -1
                signals[i] = -0.25
    
    return signals