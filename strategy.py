#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND price > 1d EMA50 AND volume > 1.5x 20-period MA.
# Short when price breaks below Donchian(20) low AND price < 1d EMA50 AND volume > 1.5x 20-period MA.
# Uses ATR(14) for dynamic stoploss via signal=0 when adverse move exceeds 2*ATR.
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.30.

name = "4h_Donchian20_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian(20) on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    
    # Volume regime: current 4h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
            
        close_val = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        upper_channel = highest_20[i]
        lower_channel = lowest_20[i]
        atr_val = atr_14[i]
        vol_spike = volume_spike[i]
        
        # Stoploss logic
        if position == 1 and close_val < entry_price - 2.0 * atr_val:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        elif position == -1 and close_val > entry_price + 2.0 * atr_val:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Entry logic
        if position == 0:
            # Long: Break above upper channel AND price > 1d EMA50 AND volume spike
            if close_val > upper_channel and close_val > ema_50_val and vol_spike:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
            # Short: Break below lower channel AND price < 1d EMA50 AND volume spike
            elif close_val < lower_channel and close_val < ema_50_val and vol_spike:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
        elif position == 1:
            # Hold long
            signals[i] = 0.30
        elif position == -1:
            # Hold short
            signals[i] = -0.30
    
    return signals