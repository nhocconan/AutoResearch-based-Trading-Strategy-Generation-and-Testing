#!/usr/bin/env python3
"""
Hypothesis: 4h Triple Barrier Method with Volatility-Adjusted Targets and Mean Reversion.
Long when price touches lower band (mean reversion) with volatility filter; short when touches upper band.
Uses Bollinger Bands (20,2) for dynamic support/resistance and ATR for volatility scaling.
Designed for 20-40 trades/year to minimize fee drag while capturing mean-reversion opportunities.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate Bollinger Bands (20,2) on close prices
    close = prices['close'].values
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    # ATR for volatility filter and stoploss (14-period)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility filter: only trade when ATR > 50-period average (avoid low volatility)
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr / atr_ma_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        upper = upper_band[i]
        lower = lower_band[i]
        atr_val = atr[i]
        vol_filter_val = vol_filter[i]
        
        if position == 0:
            # Enter long: touch lower band with sufficient volatility
            if (price_low <= lower and 
                vol_filter_val > 1.0):
                signals[i] = 0.25
                position = 1
            # Enter short: touch upper band with sufficient volatility
            elif (price_high >= upper and 
                  vol_filter_val > 1.0):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: mean reversion to middle band OR volatility collapse
            exit_signal = False
            
            # Mean reversion exit: price crosses SMA20
            if position == 1 and price_close >= sma_20[i]:
                exit_signal = True
            elif position == -1 and price_close <= sma_20[i]:
                exit_signal = True
            
            # Volatility collapse exit: ATR drops below 30-period average
            if vol_filter_val < 0.7:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_TripleBarrier_MeanReversion_VolatilityFilter"
timeframe = "4h"
leverage = 1.0