#!/usr/bin/env python3
"""
6h_ema_vwap_pullback_1d_trend_volume_v1
Hypothesis: On 6h timeframe, buy pullbacks to EMA21 when price is above VWAP (bullish intraday bias) 
and 1d trend is up (close > EMA50), with volume confirmation. Sell/short when price is below VWAP 
and 1d trend is down, with volume confirmation. Uses VWAP as dynamic support/resistance and EMA 
as trend filter. Works in both bull and bear markets by adapting to 1d trend regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ema_vwap_pullback_1d_trend_volume_v1"
timeframe = "6h"
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
    
    # Calculate VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3
    vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    
    # Align daily EMA to 6h timeframe
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 20-period volume average on 6h
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(vwap[i]) or np.isnan(ema50_6h[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below VWAP OR 1d trend turns down
            if close[i] < vwap[i] or close[i] < ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above VWAP OR 1d trend turns up
            if close[i] > vwap[i] or close[i] > ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price pulls back to VWAP from above in uptrend
            if (close[i] >= vwap[i] * 0.998 and  # Allow small tolerance
                close[i] <= vwap[i] * 1.002 and  # Near VWAP
                vol_confirm and 
                close[i] > ema50_6h[i]):  # 1d uptrend
                position = 1
                signals[i] = 0.25
            # Short: price pulls back to VWAP from below in downtrend
            elif (close[i] >= vwap[i] * 0.998 and 
                  close[i] <= vwap[i] * 1.002 and 
                  vol_confirm and 
                  close[i] < ema50_6h[i]):  # 1d downtrend
                position = -1
                signals[i] = -0.25
    
    return signals