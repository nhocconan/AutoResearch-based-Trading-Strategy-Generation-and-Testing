#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (SMMA) with 1d trend filter and volume confirmation
# Long when jaw < teeth < lips (bullish alignment) in uptrend (price > 1d EMA34) with volume > 2x average
# Short when jaw > teeth > lips (bearish alignment) in downtrend (price < 1d EMA34) with volume > 2x average
# Williams Alligator uses smoothed moving averages (SMMA) with periods 13, 8, 5 and offsets 8, 5, 3
# Trend filter: 1d EMA34 to avoid counter-trend trades
# Volume confirmation: reduces false signals
# Target: 15-30 trades/year by requiring trend alignment + Alligator alignment + volume
# Works in bull/bear: trend filter ensures we only trade with the higher timeframe trend

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Williams Alligator components (SMMA = Smoothed Moving Average)
    # Jaw: SMMA(13, 8)
    # Teeth: SMMA(8, 5)
    # Lips: SMMA(5, 3)
    
    def smma(data, period):
        """Smoothed Moving Average"""
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            # First value is simple SMA
            result[period-1] = np.mean(data[:period])
            # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Calculate Alligator lines
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Apply offsets as per Williams Alligator specification
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Align Alligator components and EMA to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = volume > 2.0 * vol_ma[i]
        
        # Williams Alligator alignment
        bullish_alignment = jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]
        bearish_alignment = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = price > ema34_aligned[i]
        downtrend = price < ema34_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Long: bullish Alligator alignment + uptrend
                if bullish_alignment and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: bearish Alligator alignment + downtrend
                elif bearish_alignment and downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Alligator alignment breaks (jaws cross teeth) or trend changes
                if not (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]) or price <= ema34_aligned[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Alligator alignment breaks or trend changes
                if not (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]) or price >= ema34_aligned[i]:
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