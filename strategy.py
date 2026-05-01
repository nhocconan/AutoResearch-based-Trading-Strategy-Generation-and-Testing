#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and 1d chop regime filter.
# Long when price breaks above Camarilla R3 level with volume > 2.0x 20-bar 1d average and 1d chop > 61.8 (range).
# Short when price breaks below Camarilla S3 level with volume confirmation and 1d chop > 61.8.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Camarilla levels derived from prior 1d completed bar (OHLC). Target: 20-50 trades/year on 4h timeframe.
# Volume spike filters low-momentum breakouts. Chop regime ensures mean-reversion edge in ranging markets.
# Works in bull (breakouts with volume) and bear (mean reversion in chop) regimes.

name = "4h_Camarilla_R3S3_Breakout_1dVolume_1dChop_v1"
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
    
    # Load 1d data ONCE before loop for volume and chop regime filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Chop Index(14) for 1d regime filter: >61.8 = range (mean revert), <38.2 = trending
    def true_range(h, l, c):
        h_l = h - l
        h_pc = np.abs(np.subtract(h, np.roll(c, 1)))
        l_pc = np.abs(np.subtract(l, np.roll(c, 1)))
        h_pc[0] = np.abs(h[0] - c[0])
        l_pc[0] = np.abs(l[0] - c[0])
        return np.maximum(h_l, np.maximum(h_pc, l_pc))
    
    tr_chop_1d = true_range(high_1d, low_1d, close_1d)
    atr_14_1d = pd.Series(tr_chop_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    denom_1d = highest_high_14_1d - lowest_low_14_1d
    chop_1d = np.where(denom_1d != 0, 100 * np.log10(atr_14_1d / denom_1d) / np.log10(14), 50.0)
    
    # Calculate 1d volume MA(20) for spike filter
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d filters to 4h
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR
    start_idx = 14
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Volume confirmation: current 4h volume > 2.0x aligned 1d volume MA(20)
        vol_confirm = volume[i] > (2.0 * vol_ma_1d_aligned[i])
        
        # Chop regime filter: only trade in range market (chop > 61.8)
        chop_filter = chop_1d_aligned[i] > 61.8
        
        if not vol_confirm or not chop_filter:
            signals[i] = 0.0 if position == 0 else signals[i-1]
            # Handle stoploss and exits even when filters fail
            if position == 1:  # Long position
                if curr_close < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
            elif position == -1:  # Short position
                if curr_close > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
            continue
        
        # Calculate Camarilla levels from previous 1d completed bar
        # Need to use the 1d bar that closed before current 4h bar
        prev_1d_idx = len(df_1d) - 1  # start with last available
        # Find the 1d bar index corresponding to current time
        # We'll use align_htf_to_ltf logic: for current 4h bar, use the last completed 1d bar
        # Since we're inside the loop, we need to get the aligned 1d OHLC values
        # Simpler approach: calculate Camarilla for each 1d bar and align
        
        # Calculate Camarilla levels for each 1d bar
        typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
        range_1d = high_1d - low_1d
        camarilla_h3 = typical_price_1d + range_1d * 1.1 / 4.0
        camarilla_l3 = typical_price_1d - range_1d * 1.1 / 4.0
        camarilla_h4 = typical_price_1d + range_1d * 1.1 / 2.0
        camarilla_l4 = typical_price_1d - range_1d * 1.1 / 2.0
        
        # Align Camarilla levels to 4h
        h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
        
        if np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or \
           np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]):
            signals[i] = 0.0 if position == 0 else signals[i-1]
            continue
        
        camarilla_r3 = h3_aligned[i]  # R3 level
        camarilla_s3 = l3_aligned[i]  # S3 level
        camarilla_r4 = h4_aligned[i]  # R4 level
        camarilla_s4 = l4_aligned[i]  # S4 level
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3
            if curr_close > camarilla_r3:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Camarilla S3
            elif curr_close < camarilla_s3:
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
            # Exit: price reaches Camarilla R4 (take profit) or re-enters S3-R3 range
            elif (curr_close >= camarilla_r4) or (camarilla_s3 <= curr_close <= camarilla_r3):
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
            # Exit: price reaches Camarilla S4 (take profit) or re-enters S3-R3 range
            elif (curr_close <= camarilla_s4) or (camarilla_s3 <= curr_close <= camarilla_r3):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals