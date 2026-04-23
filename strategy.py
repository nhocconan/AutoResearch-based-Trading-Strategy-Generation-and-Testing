#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R reversal with 1d EMA34 trend filter and volume confirmation.
Long when Williams %R crosses above -80 (oversold) AND 1d EMA34 uptrend AND volume > 1.5x 20-period average.
Short when Williams %R crosses below -20 (overbought) AND 1d EMA34 downtrend AND volume > 1.5x 20-period average.
Exit when Williams %R crosses above -20 (for longs) or below -80 (for shorts) or ATR stoploss (2.0*ATR).
Williams %R is a momentum oscillator that works well in ranging markets and captures reversals.
Designed for 12h timeframe to reduce trade frequency while capturing multi-day swings in BTC/ETH.
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
    
    # Calculate Williams %R (14-period) from 12h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Load 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation (using 12h data)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        wr = williams_r[i]
        ema34 = ema34_1d_aligned[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold) AND 1d EMA34 uptrend AND volume spike
            if (wr > -80 and williams_r[i-1] <= -80 and  # Cross above -80
                close[i] > ema34 and  # Current close above EMA34 for uptrend
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Williams %R crosses below -20 (overbought) AND 1d EMA34 downtrend AND volume spike
            elif (wr < -20 and williams_r[i-1] >= -20 and  # Cross below -20
                  close[i] < ema34 and  # Current close below EMA34 for downtrend
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Williams %R reverses (above -20 for longs, below -80 for shorts)
            if position == 1 and wr > -20:
                exit_signal = True
            elif position == -1 and wr < -80:
                exit_signal = True
            
            # ATR-based stoploss: 2.0 * ATR from entry
            if position == 1 and price < entry_price - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_1dEMA34_VolumeConfirm_ATRStop"
timeframe = "12h"
leverage = 1.0