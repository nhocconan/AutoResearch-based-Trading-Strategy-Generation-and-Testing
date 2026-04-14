#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation
# Targets: 15-25 trades/year by requiring alignment of Alligator jaws/teeth/lips with 1w EMA50
# Logic: Long when lips > teeth > jaws (bullish alignment) and price > 1w EMA50 and volume > 1.3x average
#        Short when lips < teeth < jaws (bearish alignment) and price < 1w EMA50 and volume > 1.3x average
#        Exit when alignment breaks or price crosses 1w EMA50
# Position size: 0.25 to manage drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Williams Alligator (13,8,5 SMAs with future shift)
    # Jaws: 13-period SMA shifted 8 bars forward
    # Teeth: 8-period SMA shifted 5 bars forward  
    # Lips: 5-period SMA shifted 3 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(13, n):  # Start after Alligator warmup
        # Get aligned weekly EMA50
        ema_50_i = align_htf_to_ltf(prices, df_1w, ema_50_1w)[i]
        
        # Get current Alligator values (skip if any NaN)
        jaw_i = jaw[i]
        teeth_i = teeth[i]
        lips_i = lips[i]
        
        if (np.isnan(ema_50_i) or np.isnan(jaw_i) or np.isnan(teeth_i) or 
            np.isnan(lips_i) or np.isnan(vol_ma[i])):
            continue
        
        # Volume confirmation (1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Bullish alignment: Lips > Teeth > Jaws
        bullish_align = lips_i > teeth_i > jaw_i
        # Bearish alignment: Lips < Teeth < Jaws
        bearish_align = lips_i < teeth_i < jaw_i
        
        # Long: Bullish alignment, uptrend, volume confirmation
        if position == 0 and bullish_align and close[i] > ema_50_i and volume_confirm:
            position = 1
            signals[i] = position_size
        # Short: Bearish alignment, downtrend, volume confirmation
        elif position == 0 and bearish_align and close[i] < ema_50_i and volume_confirm:
            position = -1
            signals[i] = -position_size
        # Exit: Alignment breaks or price crosses weekly EMA50
        elif position != 0:
            if position == 1 and (not bullish_align or close[i] < ema_50_i):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (not bearish_align or close[i] > ema_50_i):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0