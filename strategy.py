#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 1d Ichimoku Kijun (26) as dynamic support/resistance, 
filtered by 1d Tenkan (9) crossing above/below Kijun for trend confirmation, 
with volume spike confirmation (>1.5x 20-period average). 
Enter long when price touches Kijun from above in bullish TK cross, 
short when price touches Kijun from below in bearish TK cross. 
Exit on TK cross reversal or 2x ATR stop. 
Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag 
while capturing institutional level bounces in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Ichimoku and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Align 1d Ichimoku to 6h timeframe (wait for 1d bar to close)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    
    # Volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for stoploss (20-period)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_low = prices['low'].iloc[i]
        price_high = prices['high'].iloc[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # TK cross signals
            tk_bullish = tenkan_val > kijun_val  # Tenkan above Kijun = bullish
            tk_bearish = tenkan_val < kijun_val  # Tenkan below Kijun = bearish
            
            # Enter long: price touches Kijun from above in bullish TK cross
            if (price_low <= kijun_val * 1.002 and  # Allow 0.2% tolerance for touch
                price_close >= kijun_val and  # Price at or above Kijun
                tk_bullish and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price touches Kijun from below in bearish TK cross
            elif (price_high >= kijun_val * 0.998 and  # Allow 0.2% tolerance for touch
                  price_close <= kijun_val and  # Price at or below Kijun
                  tk_bearish and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: TK cross reversal OR ATR-based stoploss
            exit_signal = False
            
            # TK cross reversal exit
            tk_bullish = tenkan_val > kijun_val
            tk_bearish = tenkan_val < kijun_val
            
            if position == 1 and tk_bearish:  # Bullish to bearish reversal
                exit_signal = True
            elif position == -1 and tk_bullish:  # Bearish to bullish reversal
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from Kijun level)
            if position == 1:
                if price_close < kijun_val - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                if price_close > kijun_val + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_Kijun_Touch_TKCross_1dVol_ATR"
timeframe = "6h"
leverage = 1.0