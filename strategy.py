#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d volume spike and ATR-based stoploss.
- Primary timeframe: 12h for execution, HTF: 1d for volume confirmation.
- Volume confirmation: current 12h volume > 1.5 * 20-period volume MA on 1d data (aligned to 12h).
- Entry: Long when price closes above Donchian(20) upper with volume confirmation.
         Short when price closes below Donchian(20) lower with volume confirmation.
- Exit: Opposite Donchian breakout or ATR-based trailing stop (2.5 * ATR from extreme).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via breakouts, in bear via short breakdowns with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation (using same Donchian lookback)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate volume MA (20-period) on 1d
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * volume_ma_1d)
    
    # Align 1d volume spike to 12h
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Donchian channels (20-period) on 12h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # ATR for dynamic stoploss (14-period) on 12h
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = np.abs(high[0] - close[0])  # First bar: use close instead of previous close
    tr3.iloc[0] = np.abs(low[0] - close[0])
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(30, lookback, 20)  # Need enough 1d bars for volume MA and lookback for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_spike = volume_spike_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            if vol_spike:
                # Bullish breakout: price closes above upper Donchian
                if curr_close > highest_high[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    long_stop = curr_low - 2.5 * atr[i]  # Initial stop below entry
                # Bearish breakout: price closes below lower Donchian
                elif curr_close < lowest_low[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    short_stop = curr_high + 2.5 * atr[i]  # Initial stop above entry
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, curr_low - 2.5 * atr[i])
            
            # Exit conditions: opposite breakout or stoploss hit
            if curr_close < lowest_low[i] or curr_close < long_stop:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, curr_high + 2.5 * atr[i])
            
            # Exit conditions: opposite breakout or stoploss hit
            if curr_close > highest_high[i] or curr_close > short_stop:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0