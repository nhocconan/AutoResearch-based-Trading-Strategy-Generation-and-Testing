#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Daily Close Relative to Weekly EMA34 + 4h Momentum (ROC10) + Volume Spike
# In bull markets: price above weekly EMA34, ROC10 > 0, volume spike → long
# In bear markets: price below weekly EMA34, ROC10 < 0, volume spike → short
# Uses weekly EMA for trend filter (more stable than daily) and daily alignment to avoid look-ahead.
# Momentum (ROC10) confirms short-term direction. Volume spike adds conviction.
# Designed to work in both bull and bear markets by adapting to trend direction.
# Targets 15-30 trades/year with strict entry conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for EMA34 (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 on weekly close
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA34 to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Calculate 4h ROC(10) for momentum
    close = prices['close'].values
    roc = np.zeros_like(close)
    roc[10:] = (close[10:] - close[:-10]) / close[:-10] * 100
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(roc[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        roc_val = roc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Bullish conditions: price above weekly EMA34, positive ROC, volume spike
            if price > ema_val and roc_val > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Bearish conditions: price below weekly EMA34, negative ROC, volume spike
            elif price < ema_val and roc_val < 0 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: reverse signal or loss of momentum
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below weekly EMA34 or ROC turns negative
                if price < ema_val or roc_val < 0:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above weekly EMA34 or ROC turns positive
                if price > ema_val or roc_val > 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WeeklyEMA34_ROC10_Volume"
timeframe = "4h"
leverage = 1.0