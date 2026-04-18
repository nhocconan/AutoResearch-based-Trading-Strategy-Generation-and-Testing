#!/usr/bin/env python3
"""
4h Volume-Weighted RSI with 12h EMA Trend Filter
Long: VWRSI(14) < 30 + price > VWAP(20) + 12h EMA34 up
Short: VWRSI(14) > 70 + price < VWAP(20) + 12h EMA34 down
Exit: VWRSI crosses 50 or price crosses VWAP
Volume-weighted RSI reduces whipsaw in low-volume moves. VWAP acts as dynamic support/resistance.
12h EMA ensures alignment with higher timeframe trend. Designed for 40-80 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_vwap(high, low, close, volume, window):
    """Calculate Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3.0
    vwap_num = (typical_price * volume).rolling(window=window, min_periods=window).sum()
    vwap_den = volume.rolling(window=window, min_periods=window).sum()
    return vwap_num / vwap_den

def calculate_vw_rsi(close, volume, window):
    """Calculate Volume-Weighted RSI"""
    # Calculate price changes
    delta = np.diff(close, prepend=close[0])
    
    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Volume-weight the gains and losses
    vol_weighted_gains = gains * volume
    vol_weighted_losses = losses * volume
    
    # Calculate weighted averages
    avg_gain = pd.Series(vol_weighted_gains).ewm(alpha=1/window, adjust=False).mean()
    avg_loss = pd.Series(vol_weighted_losses).ewm(alpha=1/window, adjust=False).mean()
    
    # Calculate RSI
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # VWAP(20) on 4h
    vwap = calculate_vwap(high, low, close, volume, 20)
    
    # Volume-Weighted RSI(14) on 4h
    vw_rsi = calculate_vw_rsi(close, volume, 14)
    
    # Get 12h data for trend filter (EMA34)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA34 to 4h
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 12h EMA slope for trend filter
    ema_slope = np.diff(ema_34_12h_aligned, prepend=ema_34_12h_aligned[0])
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 40  # need VWAP(20), VWRSI(14), and EMA calculations
    
    for i in range(start_idx, n):
        if (np.isnan(vwap[i]) or np.isnan(vw_rsi[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(ema_slope[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: VWRSI oversold + price above VWAP + 12h EMA up
            if (vw_rsi[i] < 30 and 
                price > vwap[i] and 
                ema_slope[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: VWRSI overbought + price below VWAP + 12h EMA down
            elif (vw_rsi[i] > 70 and 
                  price < vwap[i] and 
                  ema_slope[i] < 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: VWRSI crosses 50 OR price crosses below VWAP
            if (vw_rsi[i] > 50) or (price < vwap[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: VWRSI crosses 50 OR price crosses above VWAP
            if (vw_rsi[i] < 50) or (price > vwap[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VolumeWeighted_RSI_12hEMA34"
timeframe = "4h"
leverage = 1.0