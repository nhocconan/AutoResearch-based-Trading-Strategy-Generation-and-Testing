#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data ONCE for all indicators
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on 1d
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period high/low for Donchian channel
    high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 14-period RSI on 1d
    delta = np.diff(close_1d, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    
    # Align HTF indicators to 6h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    high_20_1d_aligned = align_htf_to_ltf(prices, df_1d, high_20_1d)
    low_20_1d_aligned = align_htf_to_ltf(prices, df_1d, low_20_1d)
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Get 6h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate volume spike ratio (current volume / 20-period average)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_avg
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(high_20_1d_aligned[i]) or 
            np.isnan(low_20_1d_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_val = atr_14_1d_aligned[i]
        upper_donchian = high_20_1d_aligned[i]
        lower_donchian = low_20_1d_aligned[i]
        rsi_val = rsi_14_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        price = close[i]
        
        # Entry conditions: Donchian breakout + RSI filter + volume confirmation
        if position == 0:
            # Long: break above upper Donchian with RSI > 50 and volume spike
            if price > upper_donchian and rsi_val > 50 and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with RSI < 50 and volume spike
            elif price < lower_donchian and rsi_val < 50 and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian or RSI drops below 40
            if price < lower_donchian or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian or RSI rises above 60
            if price > upper_donchian or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_DonchianBreakout_RSI_Volume"
timeframe = "6h"
leverage = 1.0