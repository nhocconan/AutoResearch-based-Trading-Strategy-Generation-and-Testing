#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and choppiness regime filter.
# Long when price breaks above Camarilla R3 level with 1d volume > 1.5x 20-period average and CHOP > 61.8 (range).
# Short when price breaks below Camarilla S3 level with same conditions.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Camarilla levels calculated from prior completed 1d bar to avoid look-ahead.
# Volume confirmation filters low-momentum breakouts. CHOP filter ensures we only trade in ranging markets.
# Works in bull (buying range dips) and bear (selling range rallies) regimes.
# Target: 20-35 trades/year on 4h timeframe.

name = "4h_Camarilla_R3S3_Breakout_1dVolume_CHOP_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla, volume, and CHOP (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R3, S3) from prior completed day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = []
    camarilla_s3 = []
    for i in range(len(close_1d)):
        if i < 1:  # Need prior day
            camarilla_r3.append(np.nan)
            camarilla_s3.append(np.nan)
        else:
            # Camarilla levels: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
            r3 = close_1d[i-1] + 1.1 * (high_1d[i-1] - low_1d[i-1]) / 2
            s3 = close_1d[i-1] - 1.1 * (high_1d[i-1] - low_1d[i-1]) / 2
            camarilla_r3.append(r3)
            camarilla_s3.append(s3)
    
    camarilla_r3 = np.array(camarilla_r3)
    camarilla_s3 = np.array(camarilla_s3)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 1d Choppiness Index (CHOP) - 14 period
    # CHOP = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(14)
    # We'll use a simplified version: CHOP = 100 * log10(atr_sum / (hh - ll)) / log10(14)
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.maximum(high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]))], tr_1d])
    atr_1d = pd.Series(tr_1d).rolling(window=1, min_periods=1).sum().values  # True Range itself
    
    chop = np.full(len(close_1d), np.nan)
    for i in range(13, len(close_1d)):  # Need 14 periods
        atr_sum = np.sum(tr_1d[i-13:i+1])
        hh = np.max(high_1d[i-13:i+1])
        ll = np.min(low_1d[i-13:i+1])
        if hh - ll > 0:
            chop[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, CHOP, and volume
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume spike: current volume > 1.5x 1d volume average
        if vol_ma_1d_aligned[i] <= 0 or np.isnan(vol_ma_1d_aligned[i]):
            volume_spike = False
        else:
            volume_spike = curr_volume > (vol_ma_1d_aligned[i] * 1.5)
        
        # CHOP filter: only trade in ranging markets (CHOP > 61.8)
        chop_filter = chop_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla R3 breakout up AND volume spike AND chop filter
            if (curr_high > camarilla_r3_aligned[i] and 
                volume_spike and 
                chop_filter):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Camarilla S3 breakout down AND volume spike AND chop filter
            elif (curr_low < camarilla_s3_aligned[i] and 
                  volume_spike and 
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
            # Exit: price re-enters Camarilla H-L range OR CHOP drops below 38.2 (trend)
            elif (curr_low >= camarilla_s3_aligned[i] and curr_low <= camarilla_r3_aligned[i]) or \
                 (chop_aligned[i] < 38.2):
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
            # Exit: price re-enters Camarilla H-L range OR CHOP drops below 38.2 (trend)
            elif (curr_high >= camarilla_s3_aligned[i] and curr_high <= camarilla_r3_aligned[i]) or \
                 (chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals