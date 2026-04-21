#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 Trend Filter and Volume Spike
# Williams Alligator: Jaw (13-period smoothed, shifted 8), Teeth (8-period smoothed, shifted 5), Lips (5-period smoothed, shifted 3)
# Alligator is "sleeping" when lines are intertwined (range), "awakening" when lines diverge (trend)
# Long when Lips > Teeth > Jaw (bullish alignment) and price > 1d EMA34 with volume spike
# Short when Lips < Teeth < Jaw (bearish alignment) and price < 1d EMA34 with volume spike
# Volume spike (>1.5x 20-period average) confirms conviction
# Works in bull/bear: 1d EMA34 filter ensures we trade with higher timeframe trend
# Target: 15-30 trades/year by requiring Alligator alignment + EMA34 trend + volume

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate smoothed moving averages for Williams Alligator
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Smoothed Moving Average (SMMA) - similar to EMA but with different smoothing
    def smma(values, period):
        sma = np.full(len(values), np.nan)
        smma = np.full(len(values), np.nan)
        # Calculate initial SMA
        sma[period-1] = np.mean(values[:period])
        smma[period-1] = sma[period-1]
        # Calculate SMMA for remaining values
        for i in range(period, len(values)):
            smma[i] = (smma[i-1] * (period-1) + values[i]) / period
        return smma
    
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw_smma = smma(close, 13)
    jaw = np.roll(jaw_smma, 8)
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth_smma = smma(close, 8)
    teeth = np.roll(teeth_smma, 5)
    # Lips: 5-period SMMA, shifted 3 bars
    lips_smma = smma(close, 5)
    lips = np.roll(lips_smma, 3)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or 
            np.isnan(jaw[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend filter: price vs 1d EMA34
        uptrend = price > ema34_1d_aligned[i]
        downtrend = price < ema34_1d_aligned[i]
        
        # Alligator alignment
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        bullish_alignment = lips_above_teeth and teeth_above_jaw
        bearish_alignment = lips_below_teeth and teeth_below_jaw
        
        if position == 0:
            if volume_confirm:
                # Long: Bullish alignment and uptrend
                if bullish_alignment and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Bearish alignment and downtrend
                elif bearish_alignment and downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if bullish alignment breaks or trend fails
                if not bullish_alignment or not uptrend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if bearish alignment breaks or trend fails
                if not bearish_alignment or not downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0