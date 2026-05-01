#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1w trend filter.
# Long when price breaks above Donchian(20) high with volume > 1.5x 20-bar average and weekly close > weekly open (bullish week).
# Short when price breaks below Donchian(20) low with volume confirmation and weekly close < weekly open (bearish week).
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Target: 12-37 trades/year on 12h timeframe. Works in bull (breakouts with bullish weekly trend) and bear (breakouts with bearish weekly trend).

name = "12h_Donchian_20_WeeklyTrend_Volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Load 1w data ONCE before loop for weekly trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly trend: bullish if weekly close > weekly open, bearish if weekly close < weekly open
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_trend_bullish = weekly_close > weekly_open  # True for bullish week
    weekly_trend_bearish = weekly_close < weekly_open  # True for bearish week
    
    # Align weekly trend to 12h timeframe
    weekly_trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bullish.astype(float))
    weekly_trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bearish.astype(float))
    
    # Load 1d data ONCE before loop for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-day average volume for volume confirmation
    vol_ma_20d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20d)
    
    # Load 12h data ONCE before loop for Donchian levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian(20) for each 12h bar (using previous 20 completed bars)
    highest_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe (shift by 1 to use previous completed bar's levels)
    highest_high_20_aligned = align_htf_to_ltf(prices, df_12h, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_12h, lowest_low_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR and Donchian
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(weekly_trend_bullish_aligned[i]) or 
            np.isnan(weekly_trend_bearish_aligned[i]) or
            np.isnan(vol_ma_20d_aligned[i]) or
            np.isnan(highest_high_20_aligned[i]) or
            np.isnan(lowest_low_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_ma = vol_ma_20d_aligned[i]
        if vol_ma <= 0:
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 1.5)
        
        # Use previous bar's Donchian levels (already shifted by align_htf_to_ltf)
        upper_channel = highest_high_20_aligned[i]
        lower_channel = lowest_low_20_aligned[i]
        
        if np.isnan(upper_channel) or np.isnan(lower_channel):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = curr_high > upper_channel  # break above upper channel
        breakout_down = curr_low < lower_channel  # break below lower channel
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up AND volume confirmation AND bullish weekly trend
            if (breakout_up and 
                volume_confirm and 
                weekly_trend_bullish_aligned[i] > 0.5):  # treat as boolean
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Donchian breakout down AND volume confirmation AND bearish weekly trend
            elif (breakout_down and 
                  volume_confirm and 
                  weekly_trend_bearish_aligned[i] > 0.5):  # treat as boolean
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
            # Exit: price re-enters Donchian channel OR weekly trend turns bearish
            elif (curr_low >= lower_channel and curr_low <= upper_channel) or \
                 (weekly_trend_bearish_aligned[i] > 0.5):  # trend turned bearish
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
            # Exit: price re-enters Donchian channel OR weekly trend turns bullish
            elif (curr_high >= lower_channel and curr_high <= upper_channel) or \
                 (weekly_trend_bullish_aligned[i] > 0.5):  # trend turned bullish
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals