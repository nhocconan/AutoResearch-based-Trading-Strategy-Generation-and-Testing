#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot breakout with volume spike and ADX regime filter.
# Long when: price breaks above 1d Camarilla R3 level AND volume > 2.0x 20-bar average AND ADX(14) > 25
# Short when: price breaks below 1d Camarilla S3 level AND volume > 2.0x 20-bar average AND ADX(14) > 25
# Exit via ATR(20) trailing stop: long exit when price < highest_high_since_entry - 2.5 * ATR
#                      short exit when price > lowest_low_since_entry + 2.5 * ATR
# Uses 1d Camarilla for structure (proven edge), volume spike for confirmation, ADX to avoid choppy markets
# Discrete sizing 0.28 balances return and fee drag. Target: 50-150 total trades over 4 years = 12-37/year.

name = "12h_Camarilla_R3_S3_ADX25_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivots (based on previous 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar (using previous bar's OHLC)
    camarilla_r3 = np.zeros(len(close_1d))
    camarilla_s3 = np.zeros(len(close_1d))
    camarilla_r3[0] = np.nan
    camarilla_s3[0] = np.nan
    
    for i in range(1, len(close_1d)):
        # Camarilla formulas based on previous 1d bar
        high_prev = high_1d[i-1]
        low_prev = low_1d[i-1]
        close_prev = close_1d[i-1]
        range_prev = high_prev - low_prev
        
        camarilla_r3[i] = close_prev + range_prev * 1.1 / 4
        camarilla_s3[i] = close_prev - range_prev * 1.1 / 4
    
    # Align 1d Camarilla levels to 12h timeframe (completed 1d bar only)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 12h ATR(20) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 12h ADX(14) for regime filter
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr_adx = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_adx = pd.Series(tr_adx).ewm(span=14, adjust=False, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / atr_adx
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / atr_adx
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for ATR, Camarilla, ADX calculations)
    start_idx = max(20, 14) + 5  # ATR(20) + ADX(14) warmup + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(atr[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above 1d Camarilla R3 with volume spike AND strong trend (ADX > 25)
            if close[i] > camarilla_r3_aligned[i] and volume_spike[i] and adx[i] > 25:
                signals[i] = 0.28
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: price breaks below 1d Camarilla S3 with volume spike AND strong trend (ADX > 25)
            elif close[i] < camarilla_s3_aligned[i] and volume_spike[i] and adx[i] > 25:
                signals[i] = -0.28
                position = -1
                entry_bar = i
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit when price < highest_high_since_entry - 2.5 * ATR
            if close[i] < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit when price > lowest_low_since_entry + 2.5 * ATR
            if close[i] > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals