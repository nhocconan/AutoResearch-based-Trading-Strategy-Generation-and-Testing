#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above 20-period high AND close > 12h EMA50 AND volume > 1.8x 20-period average.
Short when price breaks below 20-period low AND close < 12h EMA50 AND volume > 1.8x 20-period average.
Exit via ATR trailing stop (3x ATR) or opposite Donchian breakout.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 19-50 trades/year per symbol.
Donchian channels provide structure-based breakouts that work in both trending and ranging markets.
12h EMA50 offers smooth trend filter with appropriate lag for 4h timeframe.
Volume confirmation at 1.8x ensures institutional participation in breakouts.
"""

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
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian channels (20-period) on primary timeframe
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for trailing stop (14-period)
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20, 14)  # Ensure warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND close > 12h EMA50 AND volume spike
            if (price > donchian_high[i] and 
                close[i] > ema50_12h_aligned[i] and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = price
            # Short: price breaks below Donchian low AND close < 12h EMA50 AND volume spike
            elif (price < donchian_low[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_high_since_entry = max(highest_high_since_entry, price)
                # Exit conditions
                exit_signal = False
                
                # ATR trailing stop (3x ATR)
                if price < highest_high_since_entry - 3.0 * atr_val:
                    exit_signal = True
                # Opposite Donchian breakout
                elif price < donchian_low[i]:
                    exit_signal = True
                
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                lowest_low_since_entry = min(lowest_low_since_entry, price)
                # Exit conditions
                exit_signal = False
                
                # ATR trailing stop (3x ATR)
                if price > lowest_low_since_entry + 3.0 * atr_val:
                    exit_signal = True
                # Opposite Donchian breakout
                elif price > donchian_high[i]:
                    exit_signal = True
                
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_Donchian20_12hEMA50_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0