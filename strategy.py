#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation
# Uses Williams Alligator (Jaws, Teeth, Lips) for trend identification
# Long when Lips > Teeth > Jaws with 1d uptrend and volume spike
# Short when Lips < Teeth < Jaws with 1d downtrend and volume spike
# Designed for 4h timeframe to target 20-35 trades/year per symbol.
# Williams Alligator filters out choppy markets effectively, reducing false signals.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams Alligator and trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator on 1d data
    # Jaws: SMA of median price, period 13, shift 8
    # Teeth: SMA of median price, period 8, shift 5
    # Lips: SMA of median price, period 5, shift 3
    median_price_1d = (high_1d + low_1d) / 2
    
    jaws_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    jaws_1d = np.roll(jaws_1d, 8)
    jaws_1d[:8] = np.nan
    
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    teeth_1d = np.roll(teeth_1d, 5)
    teeth_1d[:5] = np.nan
    
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    lips_1d = np.roll(lips_1d, 3)
    lips_1d[:3] = np.nan
    
    # Align Williams Alligator lines to 4h timeframe
    jaws_1d_aligned = align_htf_to_ltf(prices, df_1d, jaws_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter (20-period on 4h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if data not ready
        if (np.isnan(jaws_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaws (bullish alignment) + 1d uptrend + volume spike
            if (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaws_1d_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaws (bearish alignment) + 1d downtrend + volume spike
            elif (lips_1d_aligned[i] < teeth_1d_aligned[i] < jaws_1d_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator lines cross (trend change) or trend filter reverses
            if position == 1:
                # Exit on bearish cross or trend reversal
                if (lips_1d_aligned[i] < teeth_1d_aligned[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on bullish cross or trend reversal
                if (lips_1d_aligned[i] > teeth_1d_aligned[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0