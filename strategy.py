#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d trend filter
# Donchian channels identify key breakout levels where price exceeds recent highs/lows.
# Breakouts with volume spike indicate strong institutional participation and momentum.
# 1d EMA(50) ensures alignment with longer-term trend to avoid counter-trend trades.
# Designed for low trade frequency (20-50/year) to minimize fee drag in both bull and bear markets.
# Uses 4h timeframe as primary, with 1d HTF for trend filter and volume confirmation.

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d_s = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h Donchian channels (20-period)
    # Using pandas rolling for vectorized computation before loop
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA(50) and Donchian(20)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_ema = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        curr_dc_high = donchian_high[i]
        curr_dc_low = donchian_low[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 4h Donchian high with 1d uptrend
                if curr_close > curr_dc_high and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 4h Donchian low with 1d downtrend
                elif curr_close < curr_dc_low and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 4h Donchian low
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_dc_low:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR above entry
            elif curr_close >= entry_price + 1.5 * curr_atr:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 4h Donchian high
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_dc_high:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR below entry
            elif curr_close <= entry_price - 1.5 * curr_atr:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals