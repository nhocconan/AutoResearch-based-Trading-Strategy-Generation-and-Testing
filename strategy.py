#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume spike and ATR-based stoploss
# - Enter long when price breaks above Donchian(20) upper band AND 1d volume > 1.5x 20-period volume SMA
# - Enter short when price breaks below Donchian(20) lower band AND 1d volume > 1.5x 20-period volume SMA
# - Exit via ATR trailing stop: long stops when price < highest high since entry - 2.5*ATR, short stops when price > lowest low since entry + 2.5*ATR
# - Donchian breakouts capture momentum bursts
# - Volume confirmation ensures institutional participation
# - ATR stoploss adapts to volatility and limits drawdown
# - Target: 12-30 trades/year to minimize fee drag while capturing high-probability breakouts
# - Works in both bull and bear markets by trading breakouts in direction of volatility expansion

name = "12h_donchian_atr_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for volume confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute Donchian channels for 12h data (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute ATR for 12h data (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume SMA for 1d data (20-period)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1d volume aligned for comparison
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    for i in range(20, n):  # Start after 20-bar warmup for Donchian and ATR
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(atr[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        vol_confirm = volume_1d_aligned[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > highest_high_20[i-1]  # Break above previous upper band
        breakout_down = close[i] < lowest_low_20[i-1]  # Break below previous lower band
        
        # ATR-based trailing stop logic
        if position == 1:  # Long position
            # Update highest high since entry
            if 'entry_high' not in locals():
                entry_high = high[i]
            else:
                entry_high = max(entry_high, high[i])
            # Stop condition: price < entry_high - 2.5 * atr[i]
            if close[i] < entry_high - 2.5 * atr[i]:
                position = 0
                signals[i] = 0.0
                # Clean up entry_high for next trade
                if 'entry_high' in locals():
                    del entry_high
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Update lowest low since entry
            if 'entry_low' not in locals():
                entry_low = low[i]
            else:
                entry_low = min(entry_low, low[i])
            # Stop condition: price > entry_low + 2.5 * atr[i]
            if close[i] > entry_low + 2.5 * atr[i]:
                position = 0
                signals[i] = 0.0
                # Clean up entry_low for next trade
                if 'entry_low' in locals():
                    del entry_low
            else:
                signals[i] = -0.25  # Maintain short position
        else:
            # No position - look for breakout entries with volume confirmation
            if breakout_up and vol_confirm:
                position = 1
                entry_high = high[i]  # Initialize entry high for trailing stop
                signals[i] = 0.25
            elif breakout_down and vol_confirm:
                position = -1
                entry_low = low[i]  # Initialize entry low for trailing stop
                signals[i] = -0.25
            else:
                signals[i] = 0.0  # Stay flat
    
    return signals