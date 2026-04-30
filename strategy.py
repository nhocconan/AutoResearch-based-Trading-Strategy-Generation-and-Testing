#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w EMA(34) trend filter with Donchian(20) breakout and volume confirmation
# Donchian breakouts capture strong momentum moves, filtered by weekly trend to avoid counter-trend trades.
# Volume confirmation ensures institutional participation. Designed for very low trade frequency (<25/year)
# to minimize fee drag in both bull and bear markets. Uses 1d timeframe with 1w HTF for trend filter.

name = "1d_Donchian20_Breakout_1wTrend_VolumeSpike_v1"
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
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w_s = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # warmup for EMA(34)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_34_1w_aligned[i]
        curr_atr = atr[i]
        
        # Calculate Donchian channels (20-period)
        if i >= 20:
            donchian_high = np.max(high[i-20:i])
            donchian_low = np.min(low[i-20:i])
        else:
            donchian_high = np.max(high[:i+1])
            donchian_low = np.min(low[:i+1])
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 20-day Donchian high with 1w uptrend
                if curr_close > donchian_high and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 20-day Donchian low with 1w downtrend
                elif curr_close < donchian_low and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry price OR price breaks 20-day Donchian low
            if curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < donchian_low:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 2.5 * ATR above entry price
            elif curr_close >= entry_price + 2.5 * curr_atr:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry price OR price breaks 20-day Donchian high
            if curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > donchian_high:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 2.5 * ATR below entry price
            elif curr_close <= entry_price - 2.5 * curr_atr:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals