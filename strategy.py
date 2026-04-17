#!/usr/bin/env python3
"""
4h Donchian Breakout + 1d RSI + Volume Spike
Long: Price > Donchian High(20) + RSI(1d) > 55 + Volume > 1.5x 4h Volume SMA(20)
Short: Price < Donchian Low(20) + RSI(1d) < 45 + Volume > 1.5x 4h Volume SMA(20)
Exit: Opposite Donchian break or RSI crosses 50
Designed to capture breakouts with momentum and volume confirmation.
Target: 75-200 total trades over 4 years (19-50/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d = rsi_14_1d.fillna(50).values  # neutral when undefined
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Donchian Channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume SMA(20)
    vol_sma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # need Donchian and volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_14_1d_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(vol_sma_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_4h[i]
        rsi_val = rsi_14_1d_aligned[i]
        upper = donch_high[i]
        lower = donch_low[i]
        
        if position == 0:
            # Long: Price > Donchian High + RSI > 55 + volume spike
            if price > upper and rsi_val > 55 and vol > 1.5 * vol_sma_val:
                signals[i] = 0.25
                position = 1
            # Short: Price < Donchian Low + RSI < 45 + volume spike
            elif price < lower and rsi_val < 45 and vol > 1.5 * vol_sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price < Donchian Low or RSI < 50
            if price < lower or rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price > Donchian High or RSI > 50
            if price > upper or rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dRSI_VolumeSpike"
timeframe = "4h"
leverage = 1.0