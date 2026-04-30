#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator strategy with 1d trend filter and volume confirmation.
# Williams Alligator consists of three SMAs (Jaw=13, Teeth=8, Lips=5) with future shifts.
# Long when Lips > Teeth > Jaw (bullish alignment) and price > Lips, with 1d uptrend (close > 1d EMA34) and volume > 1.5x 20-bar avg.
# Short when Lips < Teeth < Jaw (bearish alignment) and price < Lips, with 1d downtrend (close < 1d EMA34) and volume > 1.5x 20-bar avg.
# Exit when Alligator lines cross (Lips crosses Teeth) or price crosses Jaw.
# Uses proven Alligator structure with strict volume confirmation and 1d EMA34 trend filter to limit trades (target 12-37/year).
# Timeframe: 12h, HTF: 1d as per experiment guidelines.

name = "12h_WilliamsAlligator_1dEMA34_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward  
    # Lips: 5-period SMMA shifted 3 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Alligator and EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_lips = lips[i]
        curr_teeth = teeth[i]
        curr_jaw = jaw[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator: Lips > Teeth > Jaw
            bullish_alignment = curr_lips > curr_teeth > curr_jaw
            # Bearish Alligator: Lips < Teeth < Jaw
            bearish_alignment = curr_lips < curr_teeth < curr_jaw
            
            # Long: bullish alignment, price > Lips, 1d uptrend, volume spike
            if (bullish_alignment and 
                curr_close > curr_lips and 
                curr_close > curr_ema_34_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment, price < Lips, 1d downtrend, volume spike
            elif (bearish_alignment and 
                  curr_close < curr_lips and 
                  curr_close < curr_ema_34_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit conditions: 
            # 1. Lips crosses below Teeth (Alligator weakening)
            # 2. Price crosses below Jaw (trend change)
            lips_cross_below_teeth = curr_lips < curr_teeth and lips[i-1] >= teeth[i-1] if i > 0 else False
            price_cross_below_jaw = curr_close < curr_jaw and close[i-1] >= jaw[i-1] if i > 0 else False
            
            if lips_cross_below_teeth or price_cross_below_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Lips crosses above Teeth (Alligator weakening)
            # 2. Price crosses above Jaw (trend change)
            lips_cross_above_teeth = curr_lips > curr_teeth and lips[i-1] <= teeth[i-1] if i > 0 else False
            price_cross_above_jaw = curr_close > curr_jaw and close[i-1] <= jaw[i-1] if i > 0 else False
            
            if lips_cross_above_teeth or price_cross_above_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals