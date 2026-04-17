#!/usr/bin/env python3
"""
Hypothesis: On the 4-hour timeframe, price tends to revert to the mean when it deviates significantly from the 20-period VWAP, 
especially when accompanied by extreme RSI readings and low volatility (squeeze conditions). 
We use a Bollinger Band squeeze to identify low-volatility regimes, then look for mean reversion 
when price touches the outer Bollinger Bands with RSI extremes. 
This strategy works in both bull and bear markets by capturing overextended moves that are likely to revert.
We limit trades to ~20-30 per year by requiring multiple confluence factors.
"""

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
    
    # Calculate Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    sma_20 = close_series.rolling(window=20, min_periods=20).mean()
    std_20 = close_series.rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Bollinger Band Width for squeeze detection (low volatility)
    bb_width = (upper_bb - lower_bb) / sma_20
    bb_width_sma = bb_width.rolling(window=50, min_periods=50).mean()
    squeeze_condition = bb_width < bb_width_sma * 0.8  # Bollinger Band squeeze
    
    # RSI (14)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # VWAP (approximation using typical price)
    typical_price = (high + low + close) / 3
    vwap_numerator = (typical_price * volume).cumsum()
    vwap_denominator = volume.cumsum()
    vwap = vwap_numerator / vwap_denominator
    
    # Volume moving average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(sma_20.iloc[i]) or np.isnan(std_20.iloc[i]) or 
            np.isnan(rsi.iloc[i]) or np.isnan(vwap[i]) or 
            np.isnan(volume_ma.iloc[i]) or np.isnan(squeeze_condition.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi.iloc[i]
        vol = volume[i]
        vol_ma = volume_ma.iloc[i]
        squeeze = squeeze_condition.iloc[i]
        
        if position == 0:
            # Long setup: price at lower BB, RSI oversold, volume above average, in squeeze
            if price <= lower_bb.iloc[i] and rsi_val < 30 and vol > vol_ma and squeeze:
                signals[i] = 0.25
                position = 1
            # Short setup: price at upper BB, RSI overbought, volume above average, in squeeze
            elif price >= upper_bb.iloc[i] and rsi_val > 70 and vol > vol_ma and squeeze:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to VWAP or RSI normalizes
            if price >= vwap[i] or rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to VWAP or RSI normalizes
            if price <= vwap[i] or rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_BB_Squeeze_RSI_MeanReversion_VWAP"
timeframe = "4h"
leverage = 1.0