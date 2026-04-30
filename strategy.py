#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian(20) breakout + volume confirmation + ATR filter
# Donchian breakout captures strong weekly trends; volume confirms institutional participation.
# ATR-based stoploss manages risk. Designed to work in both bull and bear markets by
# requiring volume confirmation to avoid false breakouts. Target: 30-100 total trades over 4 years.

name = "1d_Donchian20_VolumeConfirm_ATRStop_v1"
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
    
    # Load 1w data ONCE before loop for Donchian calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w Donchian channels (20-period) using prior 1w bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian upper/lower bands based on prior 20 1w bars (exclude current)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 1d timeframe (wait for completed 1w bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Calculate ATR(14) for stoploss and position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # warmup for Donchian calculation
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.5x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (1.5 * vol_ma_20)
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high_aligned[i]
        curr_donchian_low = donchian_low_aligned[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike to avoid false breakouts
            if volume_spike:
                # Bullish entry: price breaks above Donchian upper band
                if curr_close > curr_donchian_high:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below Donchian lower band
                elif curr_close < curr_donchian_low:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Exit: price falls below Donchian lower band
            elif curr_close < curr_donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Exit: price rises above Donchian upper band
            elif curr_close > curr_donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals