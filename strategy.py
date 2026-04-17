#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 1d Bollinger Bands with volume spike and ATR filter.
- Long when price breaks above upper BB(20,2) with volume > 2x 20-period 6h volume MA and ATR(14) > 0.5 * price
- Short when price breaks below lower BB(20,2) with volume > 2x 20-period 6h volume MA and ATR(14) > 0.5 * price
- Fixed position size 0.25 to limit fee churn and manage drawdown
- ATR-based trailing stop (2.0x ATR) to lock in profits
- Bollinger Bands calculated on 1d timeframe (no look-ahead)
- Designed for low trade frequency (target: 50-150 trades over 4 years) to avoid fee drag
- Works in bull markets (buying breakouts with volume and volatility confirmation) and bear markets (selling breakdowns)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Bollinger Bands (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Bollinger Bands (20,2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Align Bollinger Bands to 6h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Get 6h data for volume confirmation and ATR (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Volume average (20-period) on 6h for confirmation
    volume_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) on 6h for volatility filter and stoploss
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to 6h timeframe (primary)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr_14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper_val = upper_bb_aligned[i]
        lower_val = lower_bb_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for Bollinger Band breakouts with volume and volatility confirmation
            # Long: price breaks above upper BB + volume spike + sufficient volatility
            if price > upper_val and vol > 2.0 * vol_ma and atr_val > 0.5 * price:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.0 * atr_val
            # Short: price breaks below lower BB + volume spike + sufficient volatility
            elif price < lower_val and vol > 2.0 * vol_ma and atr_val > 0.5 * price:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_stop = entry_price + 2.0 * atr_val
        
        elif position == 1:
            # Check stoploss
            if price <= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail stop: raise stop if price moves favorably
                atr_stop = max(atr_stop, price - 1.5 * atr_val)
        
        elif position == -1:
            # Check stoploss
            if price >= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stop: lower stop if price moves favorably
                atr_stop = min(atr_stop, price + 1.5 * atr_val)
    
    return signals

name = "6h_BollingerBreakout_VolumeSpike_ATRFilter"
timeframe = "6h"
leverage = 1.0