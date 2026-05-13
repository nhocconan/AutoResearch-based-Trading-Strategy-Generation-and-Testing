#!/usr/bin/env python3
"""
4h_RSI_Bollinger_Squeeze_MeanReversion
Hypothesis: In 4h timeframe, combine RSI mean-reversion with Bollinger Bands squeeze to identify high-probability reversals in both bull and bear markets. When Bollinger Bands contract (low volatility) and RSI reaches extreme levels (<30 or >70), expect mean reversion. Use volume confirmation to filter false signals. Designed for ~30-50 trades/year to avoid fee drag.
"""

name = "4h_RSI_Bollinger_Squeeze_MeanReversion"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate RSI (14-period)
    delta = pd.Series(close).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (equivalent to EMA with alpha=1/14)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands (20-period, 2 std dev)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_stddev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_stddev * bb_std)
    lower_band = sma - (bb_stddev * bb_std)
    bb_width = upper_band - lower_band
    
    # Bollinger Band squeeze detection: width below 20-period average
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_squeeze = bb_width < bb_width_ma
    
    # Volume confirmation: volume above 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > vol_ma_20
    
    signals = np.zeros(n)
    
    for i in range(30, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(sma[i]) or np.isnan(bb_width_ma[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Mean reversion signals only during Bollinger squeeze (low volatility)
        if bb_squeeze[i]:
            # Oversold condition: RSI < 30 + volume confirmation -> long
            if rsi[i] < 30 and volume_confirm[i]:
                signals[i] = 0.25
            # Overbought condition: RSI > 70 + volume confirmation -> short
            elif rsi[i] > 70 and volume_confirm[i]:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:
            # Outside squeeze, no clear signal
            signals[i] = 0.0
    
    return signals