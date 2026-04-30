#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout with volume confirmation and 1d EMA(50) trend filter
# Donchian channels identify key support/resistance where breakouts indicate strong momentum.
# Volume confirmation ensures institutional participation. 1d EMA(50) aligns with daily trend to avoid counter-trend trades.
# Session filter (08-20 UTC) reduces noise during low-liquidity hours.
# Designed for low trade frequency (15-37/year) to minimize fee drag in both bull and bear markets.

name = "1h_Donchian20_4hTrend_VolumeSpike_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate upper and lower bands
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe (wait for completed 4h bar)
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    
    # Load 1d data ONCE before loop for EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 1h timeframe (wait for completed 1d bar)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA(50) and Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_upper = upper_20_aligned[i]
        curr_lower = lower_20_aligned[i]
        curr_ema = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: price breaks above 4h Donchian upper with 1d uptrend
                if curr_close > curr_upper and curr_close > curr_ema:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 4h Donchian lower with 1d downtrend
                elif curr_close < curr_lower and curr_close < curr_ema:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 4h Donchian lower
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_lower:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches midpoint of Donchian channel
            elif curr_close >= (curr_upper + curr_lower) / 2.0:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 4h Donchian upper
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_upper:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches midpoint of Donchian channel
            elif curr_close <= (curr_upper + curr_lower) / 2.0:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.20
    
    return signals