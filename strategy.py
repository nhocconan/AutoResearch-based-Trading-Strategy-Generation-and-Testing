#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian(20) breakout with volume confirmation and ATR-based trailing stop.
# Long when price breaks above 1w Donchian high (20-period) with volume > 1.5x 20-period average.
# Short when price breaks below 1w Donchian low (20-period) with volume > 1.5x 20-period average.
# Uses 1w HTF for Donchian channels to capture strong weekly trends while minimizing noise.
# Volume confirmation filters breakouts, ensuring only high-momentum moves trigger entries.
# ATR-based trailing stop (3.0 * ATR) protects capital during reversals and whipsaws.
# Designed for low trade frequency (~7-25/year on 1d) to minimize fee drag in ranging/bear markets.
# Works in bull markets via trend continuation and in bear markets via capturing sharp reversals.
# Focus on BTC/ETH as primary targets; avoids SOL-only bias.

name = "1d_1wDonchian20_Breakout_VolumeSpike_ATRTrail_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian high: max(high_1w, lookback=20)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Donchian low: min(low_1w, lookback=20)
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align 1w Donchian levels to 1d timeframe (wait for 1w bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Calculate ATR(14) for trailing stop on 1d
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    start_idx = 40  # warmup for Donchian(20) and ATR(14)
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_dch = donchian_high_aligned[i]
        curr_dcl = donchian_low_aligned[i]
        curr_vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            if curr_vol_spike:
                # Bullish entry: price breaks above 1w Donchian high
                if curr_close > curr_dch:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                # Bearish entry: price breaks below 1w Donchian low
                elif curr_close < curr_dcl:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest price since entry
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            # Trailing stop: 3.0 * ATR below highest price since entry
            if curr_close < highest_since_entry - 3.0 * curr_atr:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            # Trailing stop: 3.0 * ATR above lowest price since entry
            if curr_close > lowest_since_entry + 3.0 * curr_atr:
                signals[i] = 0.0
                position = 0
    
    return signals