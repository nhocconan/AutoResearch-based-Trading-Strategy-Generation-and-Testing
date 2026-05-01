#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
# Long when price breaks above Donchian(20) high with volume > 1.8x 20-bar average and weekly pivot bias bullish (price > weekly pivot).
# Short when price breaks below Donchian(20) low with volume confirmation and weekly pivot bias bearish (price < weekly pivot).
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Weekly pivot calculated from prior completed 1w bar (OHLC). Target: 12-37 trades/year on 6h timeframe.
# Volume spike filters low-momentum breakouts. Weekly pivot provides structural bias from higher timeframe.
# Works in bull (breakouts with bullish bias) and bear (breakouts with bearish bias) regimes.

name = "6h_Donchian_20_WeeklyPivot_Volume_v1"
timeframe = "6h"
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
    
    # Load 1w data ONCE before loop for weekly pivot (HTF filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Use pivot as bias filter: bullish if price > pivot, bearish if price < pivot
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR and Donchian
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or np.isnan(pivot_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.8x 20-bar average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        if vol_ma <= 0 or np.isnan(vol_ma):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 1.8)
        
        # Load 6h data ONCE before loop for Donchian levels
        df_6h = get_htf_data(prices, '6h')
        if len(df_6h) < 20:
            signals[i] = 0.0
            continue
        
        high_6h = df_6h['high'].values
        low_6h = df_6h['low'].values
        
        # Calculate Donchian(20) for each 6h bar (using previous 20 completed bars)
        highest_high_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
        lowest_low_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
        
        # Align to 6h timeframe (shift by 1 to use previous completed bar's levels)
        highest_high_20_aligned = align_htf_to_ltf(prices, df_6h, highest_high_20)
        lowest_low_20_aligned = align_htf_to_ltf(prices, df_6h, lowest_low_20)
        
        # Use previous bar's Donchian levels (already shifted by align_htf_to_ltf)
        upper_channel = highest_high_20_aligned[i]
        lower_channel = lowest_low_20_aligned[i]
        
        if np.isnan(upper_channel) or np.isnan(lower_channel):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = curr_high > upper_channel  # break above upper channel
        breakout_down = curr_low < lower_channel  # break below lower channel
        
        # Weekly pivot bias filter
        pivot_bullish = curr_close > pivot_1w_aligned[i]  # price above weekly pivot = bullish bias
        pivot_bearish = curr_close < pivot_1w_aligned[i]  # price below weekly pivot = bearish bias
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up AND volume confirmation AND bullish weekly pivot bias
            if (breakout_up and 
                volume_confirm and 
                pivot_bullish):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Donchian breakout down AND volume confirmation AND bearish weekly pivot bias
            elif (breakout_down and 
                  volume_confirm and 
                  pivot_bearish):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Donchian channel OR weekly pivot bias turns bearish
            elif (curr_low >= lower_channel and curr_low <= upper_channel) or \
                 (curr_close < pivot_1w_aligned[i]):  # bias turned bearish
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Donchian channel OR weekly pivot bias turns bullish
            elif (curr_high >= lower_channel and curr_high <= upper_channel) or \
                 (curr_close > pivot_1w_aligned[i]):  # bias turned bullish
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals