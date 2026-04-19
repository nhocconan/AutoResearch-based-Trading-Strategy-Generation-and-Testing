#!/usr/bin/env python3
# 4h_TRIX_VolumeSpike_Regime_Filter
# Hypothesis: TRIX momentum indicator with volume spike confirmation and Choppiness regime filter on 4h timeframe.
# TRIX (12-period) captures smoothed momentum; crossovers above/below zero line signal trend changes.
# Volume spike (>2x 20-period average) confirms institutional participation.
# Choppiness index (>61.8 = ranging, <38.2 = trending) filters trades to align with market regime.
# Designed for 4h to target 75-200 total trades over 4 years (19-50/year). Works in bull/bear via regime adaptation.

name = "4h_TRIX_VolumeSpike_Regime_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # TRIX (12-period): triple-smoothed EMA of percent change
    def calculate_trix(close_prices, period=12):
        # Single EMA
        ema1 = pd.Series(close_prices).ewm(span=period, adjust=False, min_periods=period).mean().values
        # Double EMA
        ema2 = pd.Series(ema1).ewm(span=period, adjust=False, min_periods=period).mean().values
        # Triple EMA
        ema3 = pd.Series(ema2).ewm(span=period, adjust=False, min_periods=period).mean().values
        # Percent change of triple EMA
        trix = np.zeros_like(close_prices)
        trix[period:] = (ema3[period:] - ema3[:-period]) / ema3[:-period] * 100
        return trix
    
    # Choppiness Index (14-period)
    def calculate_choppiness(high_prices, low_prices, close_prices, period=14):
        # True Range
        tr1 = high_prices - low_prices
        tr2 = np.abs(high_prices - np.roll(close_prices, 1))
        tr3 = np.abs(low_prices - np.roll(close_prices, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Sum of True Range over period
        tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        
        # Highest high and lowest low over period
        max_hh = pd.Series(high_prices).rolling(window=period, min_periods=period).max().values
        min_ll = pd.Series(low_prices).rolling(window=period, min_periods=period).min().values
        
        # Chop calculation
        chop = np.zeros_like(close_prices)
        mask = (max_hh - min_ll) > 0
        chop[mask] = 100 * np.log10(tr_sum[mask] / (max_hh[mask] - min_ll[mask])) / np.log10(period)
        return chop
    
    # Calculate TRIX on 4h data
    trix = calculate_trix(close, 12)
    
    # Calculate Choppiness Index on 4h data
    chop = calculate_choppiness(high, low, close, 14)
    
    # Volume confirmation: volume > 2.0 * 20-period average (stricter for fewer trades)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(chop[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: chop < 38.2 = trending (favor momentum), chop > 61.8 = ranging (favor mean reversion)
        # For TRIX momentum strategy, we prefer trending markets (chop < 38.2)
        trending_regime = chop[i] < 38.2
        
        if position == 0:
            # Long: TRIX crosses above zero with volume and trending regime
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                volume_confirm[i] and 
                trending_regime):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume and trending regime
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  volume_confirm[i] and 
                  trending_regime):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if TRIX crosses below zero or regime changes to ranging
            if (trix[i] < 0 and trix[i-1] >= 0) or (chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if TRIX crosses above zero or regime changes to ranging
            if (trix[i] > 0 and trix[i-1] <= 0) or (chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals