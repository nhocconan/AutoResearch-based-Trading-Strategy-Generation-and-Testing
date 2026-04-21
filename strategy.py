#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using Bollinger Band squeeze breakout with volume confirmation and 1d trend filter.
Bollinger Band squeeze indicates low volatility, often preceding explosive moves. We enter on breakout
of the upper/lower band with volume confirmation and 1d EMA50 trend filter. Exits on mean reversion
to the middle band or opposite band touch. Designed to capture volatility expansions in both bull
and bear markets while minimizing false breakouts with volume and trend filters.
Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Bollinger Bands on 4h (20, 2)
    close = prices['close'].values
    bb_length = 20
    bb_mult = 2.0
    
    # Calculate Bollinger Bands
    basis = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    dev = bb_mult * pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    upper = basis + dev
    lower = basis - dev
    
    # Bollinger Band width for squeeze detection
    bb_width = (upper - lower) / basis
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze_condition = bb_width < bb_width_ma  # True when in squeeze
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(basis[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        is_squeeze = squeeze_condition[i]
        
        if position == 0:
            # Enter long: break above upper band + uptrend (price > EMA50) + volume spike
            if (price_close > upper[i] and 
                price_close > ema_trend and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower band + downtrend (price < EMA50) + volume spike
            elif (price_close < lower[i] and 
                  price_close < ema_trend and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: mean reversion to middle band or opposite band touch
            if position == 1:
                if price_close < basis[i]:  # Return to mean
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25  # Hold long
            else:  # position == -1
                if price_close > basis[i]:  # Return to mean
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals

name = "4h_BollingerSqueeze_Breakout_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0