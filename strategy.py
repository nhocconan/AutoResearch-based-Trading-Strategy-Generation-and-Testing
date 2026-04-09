#!/usr/bin/env python3
# 1d_volatility_breakout_volume_v1
# Hypothesis: 1d volatility breakout with volume confirmation and weekly trend filter.
# Uses daily ATR-based breakout to capture momentum in both bull and bear markets.
# Weekly EMA trend filter ensures we only trade in the direction of the higher timeframe trend.
# Volume spike confirms institutional participation. Designed for 7-25 trades/year (30-100 over 4 years).
# Works in bull/bear markets: breakouts capture strong moves, weekly trend filter avoids counter-trend trades.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_volatility_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily ATR for volatility breakout
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate breakout levels (ATR multiplier)
    atr_mult = 2.0
    upper_breakout = np.roll(close, 1) + (atr * atr_mult)
    lower_breakout = np.roll(close, 1) - (atr * atr_mult)
    
    # Get weekly EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(atr[i]) or np.isnan(upper_breakout[i]) or 
            np.isnan(lower_breakout[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly EMA
            if close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly EMA
            if close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above upper breakout level, above weekly EMA, with volume spike
            if (close[i] > upper_breakout[i]) and (close[i] > ema_1w_aligned[i]) and vol_spike[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower breakout level, below weekly EMA, with volume spike
            elif (close[i] < lower_breakout[i]) and (close[i] < ema_1w_aligned[i]) and vol_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals