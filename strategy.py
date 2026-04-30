#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian channel breakout with volume confirmation and chop regime filter
# Donchian(20) on 12h captures intermediate-term trend structure with clear breakout levels.
# Volume confirmation ensures institutional participation. Chop filter avoids whipsaws in ranging markets.
# Designed for low trade frequency (~25-35/year) to minimize fee drag in both bull and bear markets.
# Works in bull markets via trend continuation breaks and in bear markets via breakdowns with volume.

name = "4h_Donchian20_12h_VolumeConfirm_ChopFilter_v1"
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
    
    # Calculate rolling max/min for Donchian bands
    high_roll = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (wait for completed 12h bar)
    upper_band = align_htf_to_ltf(prices, df_12h, high_roll)
    lower_band = align_htf_to_ltf(prices, df_12h, low_roll)
    
    # Calculate ATR(14) for dynamic stoploss and chop filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Chopiness Index (14-period) for regime filter
    # Chop > 61.8 = ranging market (mean revert), Chop < 38.2 = trending market (trend follow)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # warmup for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if chop indicates strong ranging market (avoid false breakouts)
        if chop[i] > 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        curr_close = close[i]
        curr_upper = upper_band[i]
        curr_lower = lower_band[i]
        curr_atr = atr[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and chop filter (trending market)
            if curr_volume_spike and chop[i] < 38.2:
                # Bullish entry: price breaks above 12h Donchian upper band
                if curr_close > curr_upper:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 12h Donchian lower band
                elif curr_close < curr_lower:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 12h Donchian lower band
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_lower:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches middle of Donchian channel
            elif curr_close >= (curr_upper + curr_lower) / 2:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 12h Donchian upper band
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_upper:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches middle of Donchian channel
            elif curr_close <= (curr_upper + curr_lower) / 2:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals