#!/usr/bin/env python3
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
    
    # Get daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ATR for volatility filter
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - df_1d['close'].shift())
    low_close = np.abs(df_1d['low'] - df_1d['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Bollinger Bands (20, 2.0)
    close_1d = df_1d['close'].values
    ma20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = ma20 + 2.0 * std20
    lower_bb = ma20 - 2.0 * std20
    
    # Align to 6h timeframe
    ma20_aligned = align_htf_to_ltf(prices, df_1d, ma20)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Bollinger Band width (volatility regime)
    bb_width = (upper_bb - lower_bb) / ma20
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ma20_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(bb_width_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        vol_filter = volume[i] > vol_ma[i]
        low_volatility = bb_width_aligned[i] < 0.05  # Bollinger Band squeeze
        
        # Mean reversion at Bollinger Bands with volume confirmation
        touch_upper = close[i] >= upper_bb_aligned[i]
        touch_lower = close[i] <= lower_bb_aligned[i]
        
        long_entry = touch_lower and vol_filter and low_volatility
        short_entry = touch_upper and vol_filter and low_volatility
        
        # Exit when price returns to middle band
        long_exit = close[i] >= ma20_aligned[i] and position == 1
        short_exit = close[i] <= ma20_aligned[i] and position == -1
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_BollingerMeanReversion_Volume_LowVol_Session"
timeframe = "6h"
leverage = 1.0