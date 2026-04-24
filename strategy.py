#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ATR(14) regime filter - ATR(14)/ATR(50) > 1.2 = high volatility (trend follow), < 0.8 = low volatility (mean revert).
- Donchian channels from 6h: upper/lower = 20-period high/low.
- Entry: Long when close breaks above upper DONCH AND ATR regime = trend follow AND volume > 1.5 * 6h volume MA(20);
         Short when close breaks below lower DONCH AND ATR regime = trend follow AND volume > 1.5 * 6h volume MA(20).
         In low volatility regime: fade at Donchian extremes (long at lower band, short at upper band).
- Exit: ATR-based trailing stop (2.5 * ATR(14)) from highest high/lowest low since entry.
- Signal size: 0.25 discrete to control fee drag and allow for scaling out.
- Uses ATR regime to adapt to market conditions - trend following in high vol, mean reversion in low vol.
  Volume confirmation ensures conviction on breakouts. Designed to work in both bull (trend follow) and bear (mean revert in low vol) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], tr_1d])  # first TR is high-low
    atr14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr50_1d = pd.Series(tr_1d).rolling(window=50, min_periods=50).mean().values
    
    # ATR regime: ratio > 1.2 = high vol (trend follow), < 0.8 = low vol (mean revert)
    atr_ratio = atr14_1d / atr50_1d
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 6h Donchian channels (20-period)
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h ATR(14) for stops
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high[0] - low[0]], tr])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for 6h timeframe
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # ATR50 needs 50, Donchian needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(donch_upper[i]) or 
            np.isnan(donch_lower[i]) or 
            np.isnan(vol_ma_6h[i]) or 
            np.isnan(atr14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr14[i]
        curr_atr_ratio = atr_ratio_aligned[i]
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = curr_volume > 1.5 * vol_ma_6h[i]
        
        # Regime classification
        is_high_vol = curr_atr_ratio > 1.2  # trend follow regime
        is_low_vol = curr_atr_ratio < 0.8   # mean revert regime
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                if is_high_vol:
                    # High volatility regime: trend following
                    # Long: Close breaks above upper Donchian
                    if curr_close > donch_upper[i]:
                        signals[i] = 0.25
                        position = 1
                        entry_price = curr_close
                        highest_since_entry = curr_close
                        lowest_since_entry = curr_close
                    # Short: Close breaks below lower Donchian
                    elif curr_close < donch_lower[i]:
                        signals[i] = -0.25
                        position = -1
                        entry_price = curr_close
                        highest_since_entry = curr_close
                        lowest_since_entry = curr_close
                elif is_low_vol:
                    # Low volatility regime: mean reversion at extremes
                    # Long: Close breaks below lower Donchian (fade the break)
                    if curr_close < donch_lower[i]:
                        signals[i] = 0.25
                        position = 1
                        entry_price = curr_close
                        highest_since_entry = curr_close
                        lowest_since_entry = curr_close
                    # Short: Close breaks above upper Donchian (fade the break)
                    elif curr_close > donch_upper[i]:
                        signals[i] = -0.25
                        position = -1
                        entry_price = curr_close
                        highest_since_entry = curr_close
                        lowest_since_entry = curr_close
        elif position == 1:
            # Long position: update highest since entry and check exit
            highest_since_entry = max(highest_since_entry, curr_high)
            lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Stoploss: 2.5 * ATR below highest high since entry
            stoploss = highest_since_entry - 2.5 * curr_atr
            if curr_close < stoploss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest since entry and check exit
            highest_since_entry = max(highest_since_entry, curr_high)
            lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Stoploss: 2.5 * ATR above lowest low since entry
            stoploss = lowest_since_entry + 2.5 * curr_atr
            if curr_close > stoploss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dATRRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0