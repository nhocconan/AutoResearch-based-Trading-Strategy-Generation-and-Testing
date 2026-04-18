#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator (13,8,5) with forward shift
    jaw_1d = pd.Series(close_1d).rolling(window=13, center=False).mean().shift(8).values  # Blue line (13-period)
    teeth_1d = pd.Series(close_1d).rolling(window=8, center=False).mean().shift(5).values   # Red line (8-period)
    lips_1d = pd.Series(close_1d).rolling(window=5, center=False).mean().shift(3).values    # Green line (5-period)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align all 1d indicators to 6h timeframe
    jaw_6h = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_6h = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_6h = align_htf_to_ltf(prices, df_1d, lips_1d)
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 6-period RSI for entry timing on 6h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_6h = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_6h[i]) or np.isnan(teeth_6h[i]) or np.isnan(lips_6h[i]) or
            np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or np.isnan(rsi_6h[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alligator = lips_6h[i] > teeth_6h[i] > jaw_6h[i]
        bearish_alligator = lips_6h[i] < teeth_6h[i] < jaw_6h[i]
        
        # Elder Ray confirmation
        strong_bull = bull_power_6h[i] > 0 and bull_power_6h[i] > bear_power_6h[i]
        strong_bear = bear_power_6h[i] < 0 and abs(bear_power_6h[i]) > bull_power_6h[i]
        
        # RSI filters for entry timing
        rsi_not_overbought = rsi_6h[i] < 70
        rsi_not_oversold = rsi_6h[i] > 30
        
        if position == 0:
            # Long: Bullish Alligator + strong bull power + RSI not overbought
            if bullish_alligator and strong_bull and rsi_not_overbought:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator + strong bear power + RSI not oversold
            elif bearish_alligator and strong_bear and rsi_not_oversold:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish Alligator OR bear power dominates
            if bearish_alligator or (bear_power_6h[i] > 0 and bear_power_6h[i] > bull_power_6h[i]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish Alligator OR bull power dominates
            if bullish_alligator or (bull_power_6h[i] > 0 and bull_power_6h[i] > bear_power_6h[i]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Alligator_RSI"
timeframe = "6h"
leverage = 1.0