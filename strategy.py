#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with volume confirmation and 1d chop regime filter.
# Long when price breaks above Camarilla R3 with volume > 1.8x 20-bar average and 1d chop > 61.8 (range).
# Short when price breaks below Camarilla S3 with volume confirmation and 1d chop > 61.8.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Camarilla levels derived from prior 1d completed candle. 4h timeframe targets 19-50 trades/year.
# Volume spike filters low-momentum breakouts. Chop regime ensures mean-reversion edge in ranging markets.
# Works in bull (breakouts with volume) and bear (mean reversion in chop) regimes.

name = "4h_Camarilla_R3S3_Volume_1dChop_v1"
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
    
    # Start after warmup for ATR and Camarilla
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
        
        # Load 1d data ONCE before loop for Camarilla levels (HTF)
        if i == start_idx:  # Only load once
            df_1d_cam = get_htf_data(prices, '1d')
            if len(df_1d_cam) < 1:
                signals[i] = 0.0
                continue
            high_1d_cam = df_1d_cam['high'].values
            low_1d_cam = df_1d_cam['low'].values
            close_1d_cam = df_1d_cam['close'].values
            
            # Calculate Camarilla levels for each 1d bar (using OHLC of that bar)
            # Camarilla R3 = close + 1.1*(high-low)/2
            # Camarilla S3 = close - 1.1*(high-low)/2
            camarilla_r3 = close_1d_cam + 1.1 * (high_1d_cam - low_1d_cam) / 2
            camarilla_s3 = close_1d_cam - 1.1 * (high_1d_cam - low_1d_cam) / 2
            
            # Align to 4h timeframe (use previous completed 1d bar's levels)
            camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d_cam, camarilla_r3)
            camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d_cam, camarilla_s3)
        
        # Use previous bar's Camarilla levels (already shifted by align_htf_to_ltf)
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        
        if np.isnan(r3_level) or np.isnan(s3_level):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions
        breakout_up = curr_high > r3_level  # break above R3
        breakout_down = curr_low < s3_level  # break below S3
        
        # Chop regime filter: only trade in range market (chop > 61.8)
        chop_filter = chop_1d_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla breakout up AND volume confirmation AND chop regime
            if (breakout_up and 
                volume_confirm and 
                chop_filter):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Camarilla breakout down AND volume confirmation AND chop regime
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
            # Exit: price re-enters Camarilla levels OR chop regime ends (trending)
            elif (curr_low >= s3_level and curr_low <= r3_level) or \
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
            # Exit: price re-enters Camarilla levels OR chop regime ends (trending)
            elif (curr_high >= s3_level and curr_high <= r3_level) or \
                 chop_1d_aligned[i] <= 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals