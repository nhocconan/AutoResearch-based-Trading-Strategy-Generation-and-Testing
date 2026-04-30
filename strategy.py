#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h HTF Donchian(20) breakout with volume confirmation and ATR-based trailing stop.
# Long when price breaks above 12h Donchian high(20) with volume spike; short when breaks below 12h Donchian low(20).
# Uses 12h HTF for structure to avoid lower timeframe noise and whipsaws.
# Volume confirmation ensures breakout legitimacy.
# ATR trailing stop adapts to volatility and reduces drawdown.
# Target: 20-40 trades/year per symbol for low fee drag and robust performance in both bull and bear markets.

name = "4h_12hDonchian20_Breakout_VolumeSpike_ATRTrail_v1"
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
    
    # Load 12h data ONCE before loop for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian high: rolling max of high over 20 periods
    donch_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of low over 20 periods
    donch_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian levels to 4h timeframe (wait for 12h bar to close)
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    
    # Calculate ATR(14) for 4h trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 2.0x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=1).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high_since_entry = 0.0  # for long trailing stop
    lowest_low_since_entry = 0.0    # for short trailing stop
    
    start_idx = 100  # warmup for ATR and Donchian
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_donch_high = donch_high_aligned[i]
        curr_donch_low = donch_low_aligned[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_50[i]
        volume_spike = curr_volume > (2.0 * curr_vol_ma)
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike to confirm breakout legitimacy
            if volume_spike:
                # Bullish entry: price breaks above 12h Donchian high
                if curr_close > curr_donch_high:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_high_since_entry = curr_close
                # Bearish entry: price breaks below 12h Donchian low
                elif curr_close < curr_donch_low:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_low_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest high since entry for trailing stop
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # ATR trailing stop: exit if price drops 2.5* ATR from highest high
            if curr_close < highest_high_since_entry - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Optional: exit if price re-enters the 12h Donchian channel (mean reversion)
            elif curr_close < curr_donch_high and curr_close > curr_donch_low:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:  # Short position
            # Update lowest low since entry for trailing stop
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # ATR trailing stop: exit if price rises 2.5* ATR from lowest low
            if curr_close > lowest_low_since_entry + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Optional: exit if price re-enters the 12h Donchian channel
            elif curr_close > curr_donch_low and curr_close < curr_donch_high:
                signals[i] = 0.0
                position = 0
    
    return signals