#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume spike confirmation.
# The Williams Alligator uses three SMAs (Jaw: 13-period, Teeth: 8-period, Lips: 5-period) to identify
# trends: when Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend.
# Combined with 1d EMA34 for higher timeframe trend alignment and volume spike (>1.5x average) for
# confirmation. Designed to capture strong trends while avoiding whipsaws in ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator SMAs
    # Jaw: 13-period SMMA (smoothed moving average), Teeth: 8-period, Lips: 5-period
    # SMMA is similar to EMA but with different smoothing; we'll use EMA as approximation
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need 34 periods for 1d EMA34 and Alligator components
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1d = ema_34_1d_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for spike detection
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        # Alligator signals: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        bullish_alligator = lips_val > teeth_val and teeth_val > jaw_val
        bearish_alligator = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Enter long: Bullish Alligator AND price > 1d EMA34 (uptrend) AND volume > 1.5x average
            if bullish_alligator and close[i] > ema_1d and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: Bearish Alligator AND price < 1d EMA34 (downtrend) AND volume > 1.5x average
            elif bearish_alligator and close[i] < ema_1d and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bearish Alligator OR trend reverses (price < 1d EMA34)
            if bearish_alligator or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bullish Alligator OR trend reverses (price > 1d EMA34)
            if bullish_alligator or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals