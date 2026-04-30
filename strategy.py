#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian channel breakout with volume confirmation and ATR-based trailing stop.
# Uses 12h HTF for Donchian(20) structure to capture medium-term trends and reduce whipsaws.
# Long when price breaks above 12h Donchian upper band with volume > 2.0x 20-period average.
# Short when price breaks below 12h Donchian lower band with volume > 2.0x 20-period average.
# Exit via trailing stop: highest high since entry minus 2.5*ATR for longs, lowest low since entry plus 2.5*ATR for shorts.
# Designed for low trade frequency (~20-40/year on 4h) to minimize fee drag while capturing strong trends.
# Works in bull markets via breakout continuation and in bear markets via shorting breakdowns.
# Uses discrete position sizing (0.0, ±0.25) to minimize churn and manage drawdown.

name = "4h_12hDonchian20_Breakout_VolumeSpike_TrailStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Donchian calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian(20) channels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper: 20-period high, lower: 20-period low
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian levels to 4h timeframe (wait for 12h bar to close)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # Calculate ATR(14) for 4h trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    start_idx = 20  # warmup for Donchian(20) and volume MA
    
    for i in range(start_idx, n):
        # Volume confirmation
        volume_spike = volume[i] > (2.0 * vol_ma_20[i]) if not np.isnan(vol_ma_20[i]) else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_upper = donchian_upper_aligned[i]
        curr_lower = donchian_lower_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if volume_spike:
                # Bullish entry: price breaks above 12h Donchian upper band
                if curr_close > curr_upper:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_high
                # Bearish entry: price breaks below 12h Donchian lower band
                elif curr_close < curr_lower:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_low
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            # Trailing stop: highest high since entry minus 2.5*ATR
            if curr_close < highest_since_entry - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            # Trailing stop: lowest low since entry plus 2.5*ATR
            if curr_close > lowest_since_entry + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
    
    return signals