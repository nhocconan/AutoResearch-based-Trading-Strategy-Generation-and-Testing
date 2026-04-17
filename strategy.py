#!/usr/bin/env python3
"""
12h_1d_WilliamsVixFix_MeanReversion
Strategy: 12-hour mean reversion using Williams Vix Fix (WVF) indicator with volume confirmation.
Long: WVF > 0.8 (fear spike) + volume > 1.5x 20-period avg + price < 12h VWAP
Short: WVF < 0.2 (complacency) + volume > 1.5x 20-period avg + price > 12h VWAP
Exit: Price crosses 12h VWAP
Position size: 0.25
Designed to capture mean reversion spikes in both bull and bear markets via fear/greed signals.
Timeframe: 12h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h VWAP for mean reversion target
    typical_price = (high + low + close) / 3.0
    vwap_num = (typical_price * volume).cumsum()
    vwap_den = volume.cumsum()
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Williams Vix Fix: measures market fear (high = fear, low = complacency)
    # WVF = ((Highest Close in period - Low) / Highest Close in period) * 100
    lookback = 22  # ~1 month of trading days
    highest_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).max()
    wvf = ((highest_close - low) / highest_close) * 100
    wvf = wvf.values  # convert to numpy array
    
    # Volume confirmation (20-period MA)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(wvf[i]) or 
            np.isnan(vwap[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Mean reversion signals
        fear_signal = wvf[i] > 0.8  # Extreme fear - potential bottom
        complacency_signal = wvf[i] < 0.2  # Extreme complacency - potential top
        
        # Price relative to VWAP for mean reversion
        price_below_vwap = close[i] < vwap[i]
        price_above_vwap = close[i] > vwap[i]
        
        if position == 0:
            # Long: fear spike + volume + price below VWAP (oversold bounce)
            if fear_signal and volume_filter and price_below_vwap:
                signals[i] = 0.25
                position = 1
            # Short: complacency spike + volume + price above VWAP (overbought fade)
            elif complacency_signal and volume_filter and price_above_vwap:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses above VWAP (mean reversion complete)
            if price_above_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below VWAP (mean reversion complete)
            if price_below_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_WilliamsVixFix_MeanReversion"
timeframe = "12h"
leverage = 1.0