#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with volume confirmation and ATR-based trend filter
# Donchian channels identify key support/resistance where institutional order flow clusters.
# Breakouts above upper or below lower band with volume spike indicate strong institutional participation.
# ATR(50) slope ensures alignment with medium-term trend to avoid counter-trend trades.
# Designed for low trade frequency (<30/year) to minimize fee drag in both bull and bear markets.
# Uses 12h timeframe as requested, with 1d HTF for Donchian levels and trend filter.

name = "12h_Donchian20_Breakout_1dATRTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Donchian calculation and ATR trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian(20) channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: 20-period high
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (wait for completed 1d bar)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Calculate 1d ATR(50) for trend filter (using ATR slope)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], tr2])
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR slope: positive = increasing volatility (trending), negative = decreasing (ranging)
    atr_slope = np.diff(atr_50, prepend=atr_50[0])
    atr_slope_aligned = align_htf_to_ltf(prices, df_1d, atr_slope)
    
    # Calculate 12h ATR(14) for dynamic stoploss
    tr1_12h = high[1:] - low[1:]
    tr2_12h = np.abs(high[1:] - close[:-1])
    tr3_12h = np.abs(low[1:] - close[:-1])
    tr_12h = np.concatenate([[np.max([tr1_12h[0], tr2_12h[0], tr3_12h[0]])], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for ATR(50)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.8x 24-period average (~2 days on 12h)
        vol_ma_24 = np.mean(volume[max(0, i-24):i])
        volume_spike = volume[i] > (1.8 * vol_ma_24)
        
        curr_close = close[i]
        curr_upper = upper_20_aligned[i]
        curr_lower = lower_20_aligned[i]
        curr_atr_slope = atr_slope_aligned[i]
        curr_atr = atr_12h[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and ATR slope > 0 (increasing volatility = trending)
            if volume_spike and curr_atr_slope > 0:
                # Bullish entry: price breaks above 1d Donchian upper band
                if curr_close > curr_upper:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 1d Donchian lower band
                elif curr_close < curr_lower:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches midpoint of Donchian channel
            elif curr_close >= (curr_upper + curr_lower) / 2:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches midpoint of Donchian channel
            elif curr_close <= (curr_upper + curr_lower) / 2:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals