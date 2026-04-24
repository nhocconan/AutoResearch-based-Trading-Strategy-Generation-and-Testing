#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 6h to target 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ATR(14) for volatility regime - only trade when ATR(14) > ATR(50) (high volatility regime).
- Donchian channels: Upper = 20-period high, Lower = 20-period low on 6h timeframe.
- Entry: Long when price breaks above 6h Donchian Upper AND volatility regime high AND volume > 1.5 * volume MA(20).
         Short when price breaks below 6h Donchian Lower AND volatility regime high AND volume > 1.5 * volume MA(20).
- Exit: ATR-based trailing stop - exit long when price < highest_high_since_entry - 2.5*ATR(14),
        exit short when price > lowest_low_since_entry + 2.5*ATR(14).
- Signal size: 0.25 discrete to balance return and drawdown.
This strategy targets high-volume breakouts during volatile regimes, which works in both bull (strong trends) and bear (panic spikes) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) and ATR(50) for volatility regime
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Get 1d data for ATR regime confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    tr1_1d = np.abs(df_1d_high[1:] - df_1d_low[:-1])
    tr2_1d = np.abs(df_1d_high[1:] - df_1d_close[:-1])
    tr3_1d = np.abs(df_1d_low[1:] - df_1d_close[:-1])
    tr_1d = np.concatenate([[np.max([tr1_1d[0], tr2_1d[0], tr3_1d[0]])], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr_1d).rolling(window=50, min_periods=50).mean().values
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 14, 50, 20)  # Need enough bars for Donchian, ATR, Vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volatility regime: trade only when 1d ATR(14) > 1d ATR(50) (high volatility)
        vol_regime = atr_14_1d_aligned[i] > atr_50_1d_aligned[i]
        vol_confirmed = curr_volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above 6h Donchian Upper AND high volatility regime AND volume confirmed
            if curr_close > high_roll[i] and vol_regime and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            # Short: Price breaks below 6h Donchian Lower AND high volatility regime AND volume confirmed
            elif curr_close < low_roll[i] and vol_regime and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: exit when price < highest_high - 2.5*ATR(14)
            if curr_close < highest_since_entry - 2.5 * atr_14[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit when price > lowest_low + 2.5*ATR(14)
            if curr_close > lowest_since_entry + 2.5 * atr_14[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dATR_Regime_VolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0