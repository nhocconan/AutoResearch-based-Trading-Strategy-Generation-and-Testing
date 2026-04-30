#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1h EMA(21) trend filter with 4h Donchian(20) breakout and volume spike confirmation.
# The 1h EMA(21) provides timely trend direction without excessive lag, while Donchian(20) captures structure breaks.
# Volume spike confirms institutional participation. Designed for moderate trade frequency (~30-60/year on 4h) to balance edge and fees.
# Works in bull markets via breakout continuation and in bear markets via mean-reversion at channel extremes.

name = "4h_Donchian20_1hEMA21_VolumeSpike_v1"
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
    
    # Load 1h data ONCE before loop for EMA(21) trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 21:
        return np.zeros(n)
    close_1h_s = pd.Series(df_1h['close'].values)
    ema_21_1h = close_1h_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_21_1h)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, lookback-1)  # warmup
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.8x 20-period average (stricter to reduce trades)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = np.mean(volume[:i]) if i > 0 else 0
        volume_spike = volume[i] > (1.8 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_21_1h_aligned[i]
        curr_atr = atr[i]
        curr_highest = highest_high[i]
        curr_lowest = lowest_low[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 4h Donchian upper with 1h uptrend
                if curr_close > curr_highest and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 4h Donchian lower with 1h downtrend
                elif curr_close < curr_lowest and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 4h Donchian lower (reversal signal)
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_lowest:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches midpoint of channel (mean reversion tendency)
            elif curr_close >= (curr_highest + curr_lowest) / 2:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 4h Donchian upper (reversal signal)
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_highest:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches midpoint of channel (mean reversion tendency)
            elif curr_close <= (curr_highest + curr_lowest) / 2:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals