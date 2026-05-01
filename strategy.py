#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and 1w chop regime filter.
# Long when price breaks above Camarilla R3 with volume > 2.0x 20-bar average and 1w chop > 61.8 (range).
# Short when price breaks below Camarilla S3 with volume confirmation and 1w chop > 61.8.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Camarilla levels derived from prior 1d session (HLC). 12h timeframe targets 12-37 trades/year.
# Volume spike filters low-momentum breakouts. Chop regime ensures mean-reversion edge in ranging markets.
# Works in bull (breakouts with volume) and bear (mean reversion in chop) regimes.

name = "12h_Camarilla_R3S3_Breakout_1dVolume_1wChop_v1"
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
    
    # Pre-compute session hours for 00-23 UTC (12h timeframe less sensitive)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Chop Index(14) for 1w regime filter: >61.8 = range (mean revert), <38.2 = trending
    def true_range(h, l, c):
        # Vectorized TR calculation avoiding roll for efficiency
        h_l = h - l
        h_pc = np.abs(np.subtract(h, np.roll(c, 1)))
        l_pc = np.abs(np.subtract(l, np.roll(c, 1)))
        # Handle first element
        h_pc[0] = np.abs(h[0] - c[0])
        l_pc[0] = np.abs(l[0] - c[0])
        return np.maximum(h_l, np.maximum(h_pc, l_pc))
    
    # Load 1w data ONCE before loop for chop regime (HTF filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr_chop_1w = true_range(high_1w, low_1w, close_1w)
    atr_14_1w = pd.Series(tr_chop_1w).rolling(window=14, min_periods=14).sum().values
    highest_high_14_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_14_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    denom_1w = highest_high_14_1w - lowest_low_14_1w
    # Avoid division by zero
    chop_1w = np.where(denom_1w != 0, 100 * np.log10(atr_14_1w / denom_1w) / np.log10(14), 50.0)
    
    # Align 1w chop to 12h
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar (based on previous day's HLC)
    camarilla_r3 = np.full(len(high_1d), np.nan)
    camarilla_s3 = np.full(len(high_1d), np.nan)
    for j in range(1, len(high_1d)):
        phigh = high_1d[j-1]
        plow = low_1d[j-1]
        pclose = close_1d[j-1]
        range_ = phigh - plow
        camarilla_r3[j] = pclose + range_ * 1.1 / 4
        camarilla_s3[j] = pclose - range_ * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Session filter: optional for 12h, but keep for consistency
        if not (0 <= hours[i] <= 23):  # always true, kept for structure
            signals[i] = 0.0
            continue
        
        if (np.isnan(atr[i]) or np.isnan(chop_1w_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-bar average (stricter for fewer trades)
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        if vol_ma <= 0 or np.isnan(vol_ma):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 2.0)
        
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        
        # Camarilla breakout conditions
        breakout_up = curr_high > r3  # break above R3
        breakout_down = curr_low < s3  # break below S3
        
        # Chop regime filter: only trade in range market (chop > 61.8)
        chop_filter = chop_1w_aligned[i] > 61.8
        
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
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Camarilla range (between S3 and R3) OR chop regime ends (trending)
            elif (curr_low >= s3 and curr_low <= r3) or \
                 chop_1w_aligned[i] <= 61.8:
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
            # Exit: price re-enters Camarilla range (between S3 and R3) OR chop regime ends (trending)
            elif (curr_high >= s3 and curr_high <= r3) or \
                 chop_1w_aligned[i] <= 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals