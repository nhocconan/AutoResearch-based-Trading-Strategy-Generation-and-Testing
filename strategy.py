#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla H4/L4 breakout with volume confirmation in low volatility regimes
    # Uses 1d Camarilla levels as institutional support/resistance
    # Volume > 2.0x 20-period MA confirms institutional participation
    # ATR(14) < ATR(50) filter ensures low volatility breakouts (avoid fakeouts)
    # Discrete sizing 0.25 to minimize fee churn. Target: 20-40 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla H4 and L4 levels (stronger breakout levels)
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align to 4h timeframe (use previous day's levels)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: current volume > 2.0x 20-period MA
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    vol_ratio = np.full(n, np.nan)
    for i in range(20, n):
        if vol_ma_20[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma_20[i]
        else:
            vol_ratio[i] = 1.0
    
    # Volatility filter: ATR(14) < ATR(50) = low volatility environment
    # Calculate True Range
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # ATR(14)
    atr_14 = np.full(n, np.nan)
    for i in range(14, n):
        atr_14[i] = np.mean(tr[i-14:i])
    
    # ATR(50)
    atr_50 = np.full(n, np.nan)
    for i in range(50, n):
        atr_50[i] = np.mean(tr[i-50:i])
    
    # Low volatility condition
    low_vol = np.full(n, False)
    for i in range(50, n):
        if not np.isnan(atr_14[i]) and not np.isnan(atr_50[i]) and atr_50[i] > 0:
            low_vol[i] = atr_14[i] < atr_50[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions with volume confirmation in low volatility
        breakout_up = close[i] > camarilla_h4_aligned[i]
        breakout_down = close[i] < camarilla_l4_aligned[i]
        
        # Entry conditions: breakout with volume confirmation in low volatility
        long_entry = breakout_up and (vol_ratio[i] > 2.0) and low_vol[i]
        short_entry = breakout_down and (vol_ratio[i] > 2.0) and low_vol[i]
        
        # Exit conditions: price returns to 1d close (intraday mean reversion)
        long_exit = close[i] < close_1d[-1] if len(close_1d) > 0 else False
        short_exit = close[i] > close_1d[-1] if len(close_1d) > 0 else False
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_vol_lowvol_v1"
timeframe = "4h"
leverage = 1.0