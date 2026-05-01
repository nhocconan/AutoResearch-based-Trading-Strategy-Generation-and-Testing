#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter, volume confirmation, and ATR(14) stoploss.
# Long when price breaks above Donchian upper AND price > 1w EMA50 AND volume > 1.5x 1d volume median.
# Short when price breaks below Donchian lower AND price < 1w EMA50 AND volume > 1.5x 1d volume median.
# Uses discrete sizing 0.25. ATR stoploss: signal→0 when price moves against position by 2.0*ATR.
# Target: 10-25 trades/year on 1d timeframe (40-100 total over 4 years) to minimize fee drag.
# Donchian provides clear structure, 1w EMA50 offers smooth trend filter, volume confirms momentum.
# This combination has shown strong test performance in DB for SOL/ETH with proper filtering.
# Works in both bull and bear markets: trend filter prevents counter-trend entries, volume confirms strength.

name = "1d_Donchian20_Breakout_1wEMA50_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian(20) from previous 1d bar (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous 20-period high/low for Donchian channels (shifted by 1 for no look-ahead)
    donchian_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 1w EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d volume median (30-period for stability)
    vol_median_1d = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_median_1d[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 1d volume median
        if vol_median_1d[i] <= 0 or np.isnan(vol_median_1d[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_1d[i] * 1.5)
        
        # Trend filter: price vs 1w EMA50
        uptrend = curr_close > ema_50_1w_aligned[i]
        downtrend = curr_close < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above Donchian upper AND uptrend AND volume confirmation
            if (curr_high > donchian_high_aligned[i] and 
                uptrend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Break below Donchian lower AND downtrend AND volume confirmation
            elif (curr_low < donchian_low_aligned[i] and 
                  downtrend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian lower OR trend turns down
            elif (curr_low < donchian_low_aligned[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian upper OR trend turns up
            elif (curr_high > donchian_high_aligned[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals