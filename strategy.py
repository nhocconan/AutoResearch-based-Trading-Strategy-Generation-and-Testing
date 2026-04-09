#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Supertrend + volume confirmation
# - Primary signal: Elder Ray Bull/Bear Power divergence on 6h with 1d Supertrend filter
# - Bull Power = High - EMA(13), Bear Power = Low - EMA(13) 
# - Long: Bull Power rising (2-bar momentum) + Bear Power < 0 + price > EMA(13) + 1d Supertrend up
# - Short: Bear Power falling (2-bar momentum) + Bull Power < 0 + price < EMA(13) + 1d Supertrend down
# - Volume confirmation: 6h volume > 1.3x 20-period average
# - Works in bull/bear: Supertrend filters counter-trend trades, Elder Ray captures momentum exhaustion
# - Position size: 0.25 discrete level
# - Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_1d_elderray_supertrend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d Supertrend
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ATR(10)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr_3))
    tr_1d[0] = tr1[0]
    atr_10 = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend parameters
    atr_mult = 3.0
    upperband = (high_1d + low_1d) / 2 + atr_mult * atr_10
    lowerband = (high_1d + low_1d) / 2 - atr_mult * atr_10
    
    # Initialize Supertrend
    supertrend = np.full_like(close_1d, np.nan)
    direction = np.full_like(close_1d, 1)  # 1=up, -1=down
    
    for i in range(1, len(close_1d)):
        if np.isnan(supertrend[i-1]):
            supertrend[i] = lowerband[i]
            direction[i] = 1
            continue
            
        if close_1d[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_1d[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
        if direction[i] == 1:
            supertrend[i] = max(lowerband[i], supertrend[i-1])
        else:
            supertrend[i] = min(upperband[i], supertrend[i-1])
    
    supertrend_dir = align_htf_to_ltf(prices, df_1d, direction)
    
    # Pre-compute 6h Elder Ray components
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # EMA(13) for Elder Ray
    ema_13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA(13)
    bull_power = high_6h - ema_13
    # Bear Power = Low - EMA(13)
    bear_power = low_6h - ema_13
    
    # Bull/Bear Power momentum (2-bar change)
    bull_power_mom = bull_power - np.roll(bull_power, 2)
    bear_power_mom = bear_power - np.roll(bear_power, 2)
    bull_power_mom[:2] = 0
    bear_power_mom[:2] = 0
    
    # Volume confirmation
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_6h > (1.3 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bull_power_mom[i]) or np.isnan(bear_power_mom[i]) or
            np.isnan(supertrend_dir[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bear Power turns positive OR Supertrend flips down
            if bear_power[i] >= 0 or supertrend_dir[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power turns positive OR Supertrend flips up
            if bull_power[i] >= 0 or supertrend_dir[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray divergence with volume and Supertrend confirmation
            # Long: Bull Power rising momentum + Bear Power negative + price > EMA + Supertrend up
            if (bull_power_mom[i] > 0 and bear_power[i] < 0 and 
                close_6h[i] > ema_13[i] and supertrend_dir[i] == 1 and volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: Bear Power falling momentum + Bull Power negative + price < EMA + Supertrend down
            elif (bear_power_mom[i] < 0 and bull_power[i] < 0 and 
                  close_6h[i] < ema_13[i] and supertrend_dir[i] == -1 and volume_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals