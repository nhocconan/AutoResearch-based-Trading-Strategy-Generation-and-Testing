#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and choppiness regime filter.
# Long when price breaks above Camarilla R3 with volume > 2.0x 20-period volume average and CHOP > 61.8 (range regime).
# Short when price breaks below Camarilla S3 with volume confirmation and CHOP > 61.8.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Camarilla levels calculated from prior completed 1d bar to avoid look-ahead.
# Volume confirmation filters low-momentum breakouts. Choppiness regime ensures trades occur in ranging markets.
# Works in bull (mean reversion in range) and bear (mean reversion in range) markets by targeting reversals at extreme levels.
# Target: 12-30 trades/year on 12h timeframe.

name = "12h_Camarilla_R3S3_Breakout_1dVolume_Chop_v1"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop for Camarilla, volume, and choppiness (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R3, S3) from prior completed 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = high_1d - low_1d
    camarilla_r3 = close_1d + camarilla_range * 1.1 / 4
    camarilla_s3 = close_1d - camarilla_range * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Choppiness Index (14) on 1d
    # CHOP = 100 * log10(sum(ATR(14) over 14 periods) / (log10(14) * (highest high - lowest low over 14 periods)))
    # We'll use a simplified version: CHOP = 100 * log10(sum(tr) / (log10(14) * (hh - ll)))
    # where tr is true range, hh is highest high, ll is lowest low over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr_14 / (np.log10(14) * (highest_high_14 - lowest_low_14)))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Camarilla, volume, and chop
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 1d volume average
        if vol_ma_1d_aligned[i] <= 0 or np.isnan(vol_ma_1d_aligned[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma_1d_aligned[i] * 2.0)
        
        # Regime filter: CHOP > 61.8 indicates ranging market (good for mean reversion)
        ranging_market = chop_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla S3 breakout up (price > S3) with volume confirmation and ranging market
            # Note: In Camarilla, S3 is support, R3 is resistance. We mean revert at extremes.
            if (curr_close > camarilla_s3_aligned[i] and 
                volume_confirm and 
                ranging_market):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Camarilla R3 breakout down (price < R3) with volume confirmation and ranging market
            elif (curr_close < camarilla_r3_aligned[i] and 
                  volume_confirm and 
                  ranging_market):
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
            # Exit: price reaches Camarilla R3 (resistance) or chop regime ends
            elif (curr_close >= camarilla_r3_aligned[i]) or \
                 (chop_aligned[i] <= 61.8):  # regime change to trending
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
            # Exit: price reaches Camarilla S3 (support) or chop regime ends
            elif (curr_close <= camarilla_s3_aligned[i]) or \
                 (chop_aligned[i] <= 61.8):  # regime change to trending
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals