#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trailing stop
# Uses discrete sizing 0.25 to balance profit and fee drag. Target: 75-200 total trades over 4 years (19-50/year).
# Donchian channels provide clear breakout levels; volume confirms institutional participation.
# ATR trailing stop adapts to volatility and manages risk in both bull and bear markets.

name = "4h_Donchian20_VolumeSpike_ATRTrail_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Donchian channels (20-period)
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.8x 20-period average (moderate threshold)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    # ATR for trailing stop (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    
    start_idx = max(20, 20, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr_14[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_h = donchian_h[i]
        curr_donchian_l = donchian_l[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr_14[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Donchian breakout
            if curr_volume_spike:
                # Bullish: Close breaks above upper Donchian
                if curr_close > curr_donchian_h:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_high = curr_high
                # Bearish: Close breaks below lower Donchian
                elif curr_close < curr_donchian_l:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_low = curr_low
        
        elif position == 1:  # Long position
            # Update highest high for trailing stop
            highest_high = max(highest_high, curr_high)
            # ATR trailing stop: 2.5 * ATR below highest high
            stop_loss = highest_high - 2.5 * curr_atr
            # Exit: Stoploss hit OR close drops below lower Donchian (mean reversion)
            if curr_low <= stop_loss or curr_close < curr_donchian_l:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low for trailing stop
            lowest_low = min(lowest_low, curr_low)
            # ATR trailing stop: 2.5 * ATR above lowest low
            stop_loss = lowest_low + 2.5 * curr_atr
            # Exit: Stoploss hit OR close rises above upper Donchian (mean reversion)
            if curr_high >= stop_loss or curr_close > curr_donchian_h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals