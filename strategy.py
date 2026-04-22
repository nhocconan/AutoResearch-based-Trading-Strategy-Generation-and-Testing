#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams Alligator + 12h trend filter + volume confirmation
    # Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend direction.
    # Enter when price crosses above/below Teeth with Jaw/Teeth/Lips aligned.
    # 12h EMA50 filters long-term trend; volume spike confirms momentum.
    # Designed for 15-35 trades/year to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator components (6-period base)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    # Median price for Alligator calculation
    median_price = (high + low) / 2.0
    
    # Jaw (13-period SMMA of median)
    jaw = pd.Series(median_price).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    # Teeth (8-period SMMA of median)
    teeth = pd.Series(median_price).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    # Lips (5-period SMMA of median)
    lips = pd.Series(median_price).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw (alligator opening up)
            bullish_aligned = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Bearish alignment: Lips < Teeth < Jaw (alligator opening down)
            bearish_aligned = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long: Price crosses above Teeth with bullish alignment + volume + price above 12h EMA50
            if close[i] > teeth[i] and bullish_aligned and vol_spike[i] and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below Teeth with bearish alignment + volume + price below 12h EMA50
            elif close[i] < teeth[i] and bearish_aligned and vol_spike[i] and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses Jaw or trend reversal vs 12h EMA50
            if position == 1:
                if close[i] < jaw[i] or close[i] < ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > jaw[i] or close[i] > ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Williams_Alligator_12hEMA50_Volume_Session_v1"
timeframe = "6h"
leverage = 1.0