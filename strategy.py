#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d chop regime filter.
# Long when price breaks above Donchian(20) high with volume > 1.8x 20-bar average and 1d chop > 61.8 (range).
# Short when price breaks below Donchian(20) low with volume confirmation and 1d chop > 61.8.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Donchian levels derived from prior 20 completed 4h bars. 4h timeframe targets 19-50 trades/year.
# Volume spike filters low-momentum breakouts. Chop regime ensures mean-reversion edge in ranging markets.
# Works in bull (breakouts with volume) and bear (mean reversion in chop) regimes.

name = "4h_Donchian_20_Volume_1dChop_v3"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Chop Index(14) for 1d regime filter: >61.8 = range (mean revert), <38.2 = trending
    def true_range(h, l, c):
        # Vectorized TR calculation avoiding roll for efficiency
        h_l = h - l
        h_pc = np.abs(np.subtract(h, np.roll(c, 1)))
        l_pc = np.abs(np.subtract(l, np.roll(c, 1)))
        # Handle first element
        h_pc[0] = np.abs(h[0] - c[0])
        l_pc[0] = np.abs(l[0] - c[0])
        return np.maximum(h_l, np.maximum(h_pc, l_pc))
    
    # Load 1d data ONCE before loop for chop regime (HTF filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_chop_1d = true_range(high_1d, low_1d, close_1d)
    atr_14_1d = pd.Series(tr_chop_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    denom_1d = highest_high_14_1d - lowest_low_14_1d
    # Avoid division by zero
    chop_1d = np.where(denom_1d != 0, 100 * np.log10(atr_14_1d / denom_1d) / np.log10(14), 50.0)
    
    # Align 1d chop to 4h
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR and Donchian
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or np.isnan(chop_1d_aligned[i])):
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
        
        # Calculate Donchian levels from prior 20 completed 4h bars
        # Need 4h OHLC from previous completed 4h bars
        df_4h = get_htf_data(prices, '4h')
        if len(df_4h) < 20:
            signals[i] = 0.0
            continue
        
        # Align 4h data to get prior completed 4h bar's Donchian levels
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # Calculate Donchian(20) for each 4h bar (using previous 20 completed bars)
        highest_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
        lowest_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
        
        # Align to 4h timeframe (shift by 1 to use previous completed bar's levels)
        highest_high_20_aligned = align_htf_to_ltf(prices, df_4h, highest_high_20)
        lowest_low_20_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_20)
        
        # Use previous bar's Donchian levels (already shifted by align_htf_to_ltf)
        upper_channel = highest_high_20_aligned[i]
        lower_channel = lowest_low_20_aligned[i]
        
        if np.isnan(upper_channel) or np.isnan(lower_channel):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = curr_high > upper_channel  # break above upper channel
        breakout_down = curr_low < lower_channel  # break below lower channel
        
        # Chop regime filter: only trade in range market (chop > 61.8)
        chop_filter = chop_1d_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up AND volume confirmation AND chop regime
            if (breakout_up and 
                volume_confirm and 
                chop_filter):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Donchian breakout down AND volume confirmation AND chop regime
            elif (breakout_down and 
                  volume_confirm and 
                  chop_filter):
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
            # Exit: price re-enters Donchian channel OR chop regime ends (trending)
            elif (curr_low >= lower_channel and curr_low <= upper_channel) or \
                 chop_1d_aligned[i] <= 61.8:
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
            # Exit: price re-enters Donchian channel OR chop regime ends (trending)
            elif (curr_high >= lower_channel and curr_high <= upper_channel) or \
                 chop_1d_aligned[i] <= 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals