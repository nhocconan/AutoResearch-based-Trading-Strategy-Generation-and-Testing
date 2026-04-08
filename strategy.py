#!/usr/bin/env python3
# 1d_volume_momentum_breakout_v1
# Hypothesis: On daily timeframe, capture breakouts from volatility contractions using volume-weighted momentum. 
# Uses Bollinger Band squeeze (low volatility) followed by expansion with volume confirmation and momentum alignment.
# Works in bull/bear by trading breakouts in direction of momentum. Low trade frequency (~10-20/year) minimizes fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_volume_momentum_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * bb_std_dev)
    lower_band = sma - (bb_std * bb_std_dev)
    bb_width = (upper_band - lower_band) / sma
    
    # Bollinger Band Squeeze detection (low volatility)
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze = bb_width < bb_width_ma * 0.8  # Bollinger Bands contracted
    
    # Momentum indicator (Rate of Change)
    roc_period = 10
    roc = np.zeros(n)
    roc[roc_period:] = (close[roc_period:] - close[:-roc_period]) / close[:-roc_period] * 100
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > vol_ma * 1.5  # Volume 50% above average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(squeeze[i]) or np.isnan(roc[i]) or np.isnan(volume_surge[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price returns to middle Bollinger Band OR momentum fades
            if close[i] <= sma[i] or roc[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to middle Bollinger Band OR momentum fades
            if close[i] >= sma[i] or roc[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Entry conditions: Bollinger Band breakout with volume surge and momentum alignment
            if squeeze[i-1] and not squeeze[i]:  # Just exited squeeze
                # Long breakout: Price breaks above upper band with volume and positive momentum
                if (close[i] > upper_band[i] and 
                    volume_surge[i] and 
                    roc[i] > 0):
                    position = 1
                    signals[i] = 0.25
                # Short breakout: Price breaks below lower band with volume and negative momentum
                elif (close[i] < lower_band[i] and 
                      volume_surge[i] and 
                      roc[i] < 0):
                    position = -1
                    signals[i] = -0.25
    
    return signals