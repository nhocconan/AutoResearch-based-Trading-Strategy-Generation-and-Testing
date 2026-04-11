#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and ATR-based stoploss
# - Enter long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period volume SMA
# - Enter short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period volume SMA
# - Exit via ATR trailing stop: signal=0 when long position hits highest_high - 2*ATR or short hits lowest_low + 2*ATR
# - Donchian channels provide objective breakout levels
# - Volume confirmation filters false breakouts
# - ATR stoploss manages risk without look-ahead
# - Target: 20-50 trades/year to minimize fee drag while capturing strong trends

name = "4h_1d_donchian_volume_atr_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Load 1d data ONCE before loop for volume confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute Donchian(20) for 4h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Pre-compute ATR(14) for 4h data
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Pre-compute volume SMA for 1d data (20-period)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    for i in range(lookback, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(atr[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        vol_confirm = volume_1d_aligned[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Donchian breakout signals
        breakout_high = close[i] > highest_high[i-1]  # Break above previous period's high
        breakout_low = close[i] < lowest_low[i-1]     # Break below previous period's low
        
        # Update highest/lowest since entry for trailing stop
        if position == 1:  # Long position
            highest_high_since_entry = max(highest_high_since_entry, high[i])
        elif position == -1:  # Short position
            lowest_low_since_entry = min(lowest_low_since_entry, low[i]) if lowest_low_since_entry != 0 else low[i]
        elif position == 0:  # Flat - reset tracking
            highest_high_since_entry = 0.0
            lowest_low_since_entry = 0.0
        
        # ATR trailing stop conditions
        stop_long = position == 1 and highest_high_since_entry > 0 and close[i] < (highest_high_since_entry - 2.0 * atr[i])
        stop_short = position == -1 and lowest_low_since_entry > 0 and close[i] > (lowest_low_since_entry + 2.0 * atr[i])
        
        # Trading logic
        if vol_confirm:
            # Long entry: Donchian breakout high
            if breakout_high and position != 1:
                position = 1
                highest_high_since_entry = high[i]  # Initialize tracking
                signals[i] = 0.30
            # Short entry: Donchian breakout low
            elif breakout_low and position != -1:
                position = -1
                lowest_low_since_entry = low[i]  # Initialize tracking
                signals[i] = -0.30
            # Exit conditions
            elif stop_long or stop_short:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                # Maintain current position
                signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
        else:
            # No volume confirmation: exit any position
            if position != 0:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals