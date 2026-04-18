#!/usr/bin/env python3
"""
4h Williams Alligator with 1d Trend Filter and Volume Confirmation
Hypothesis: Williams Alligator (three SMAs) identifies trend direction and momentum.
In trending markets, the jaws (13-period SMA), teeth (8-period SMA), and lips (5-period SMA)
align and diverge. We use 1d EMA50 for higher timeframe trend filter to avoid counter-trend trades,
and enter on 4h when the Alligator shows strong alignment with volume confirmation.
This strategy targets 20-30 trades/year to minimize fee drag while capturing
strong trending moves. The Alligator's convergence/divergence provides clear
entry/exit signals, while the 1d trend filter ensures we trade with the
dominant higher timeframe momentum in both bull and bear markets.
"""

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
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator components on 4h
    # Jaws: 13-period SMMA (smoothed moving average)
    # Teeth: 8-period SMMA
    # Lips: 5-period SMMA
    # Using EMA as approximation for SMMA for simplicity
    jaws = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Alligator alignment metrics
    # When jaws > teeth > lips: strong uptrend
    # When jaws < teeth < lips: strong downtrend
    # When intertwined: ranging/no trend
    jaws_above_teeth = jaws > teeth
    teeth_above_lips = teeth > lips
    jaws_below_teeth = jaws < teeth
    teeth_below_lips = teeth < lips
    
    # Strong trend conditions
    strong_uptrend = jaws_above_teeth & teeth_above_lips
    strong_downtrend = jaws_below_teeth & teeth_below_lips
    
    # Volatility filter using ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume filter: current volume > 1.3x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(jaws[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        trend = ema50_1d_aligned[i]
        strong_up = strong_uptrend[i]
        strong_down = strong_downtrend[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: strong uptrend alignment, price above trend, with volume
            if strong_up and price > trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: strong downtrend alignment, price below trend, with volume
            elif strong_down and price < trend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if trend weakens or price crosses below teeth
            if not strong_up or price < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if trend weakens or price crosses above teeth
            if not strong_down or price > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0