#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume spike
# Williams Alligator uses SMAs (13,8,5) with future shifts: Jaw(13), Teeth(8), Lips(5)
# Long when Lips > Teeth > Jaw (bullish alignment) + 1d uptrend + volume spike
# Short when Lips < Teeth < Jaw (bearish alignment) + 1d downtrend + volume spike
# The Alligator's alignment acts as a trend filter, reducing false signals
# Designed for 12h timeframe to target 12-37 trades/year per symbol.
# Works in bull markets (captures trends) and bear markets (avoids whipsaws via trend filter)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for higher timeframe trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator components (using 12h data)
    # Jaw: 13-period SMMA shifted 8 bars ahead
    # Teeth: 8-period SMMA shifted 5 bars ahead  
    # Lips: 5-period SMMA shifted 3 bars ahead
    # Using SMA as approximation for SMMA (Williams uses SMMA but SMA is similar)
    ma13 = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    ma8 = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    ma5 = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Shift to align with Alligator's future-looking nature
    jaw = np.roll(ma13, 8)   # 13-period shifted 8 bars
    teeth = np.roll(ma8, 5)  # 8-period shifted 5 bars
    lips = np.roll(ma5, 3)   # 5-period shifted 3 bars
    
    # Fill NaN from rolling and shifting
    jaw[:13+8] = np.nan
    teeth[:8+5] = np.nan
    lips[:5+3] = np.nan
    
    # Volume spike filter (20-period on 12h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + 1d uptrend + volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + 1d downtrend + volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: alignment breaks or trend reversal
            if position == 1:
                # Exit on bearish alignment or trend reversal
                if (lips[i] < teeth[i] or teeth[i] < jaw[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on bullish alignment or trend reversal
                if (lips[i] > teeth[i] or teeth[i] > jaw[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0