#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# - Long when price breaks above Camarilla H3 level + 1d volume > 1.8x 20-period volume SMA + CHOP > 61.8 (range market)
# - Short when price breaks below Camarilla L3 level + 1d volume > 1.8x 20-period volume SMA + CHOP > 61.8 (range market)
# - Exit: price returns to opposite Camarilla level (L3 for long, H3 for short)
# - Position sizing: 0.25 discrete level
# - Works in bull/bear: Camarilla levels adapt to volatility, volume confirms institutional interest, CHOP filter avoids trending markets where breakouts fail
# - 4h timeframe targets 20-50 trades/year with strict entry conditions

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.125*(high-low)
    #           L3 = close - 1.125*(high-low), L4 = close - 1.5*(high-low)
    range_1d = df_1d['high'] - df_1d['low']
    camarilla_h3 = df_1d['close'] + 1.125 * range_1d
    camarilla_l3 = df_1d['close'] - 1.125 * range_1d
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3.values)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3.values)
    
    # Calculate 1d volume SMA(20) for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate Choppiness Index on 1d (14-period)
    # True Range
    tr1 = np.maximum(df_1d['high'] - df_1d['low'], 
                     np.maximum(np.abs(df_1d['high'] - np.roll(df_1d['close'], 1)), 
                                np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))))
    tr1[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = df_1d['high'].rolling(window=14, min_periods=14).max().values
    ll = df_1d['low'].rolling(window=14, min_periods=14).min().values
    # Choppiness Index: 100 * log10(sum(tr1)/(hh-ll)) / log10(14)
    chop_raw = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
    chop = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.8x 20-period SMA (spike)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)
        vol_confirm = vol_1d_current[i] > 1.8 * volume_sma_20_1d_aligned[i]
        
        # Regime filter: CHOP > 61.8 indicates ranging market (good for mean reversion breakouts)
        ranging_market = chop[i] > 61.8
        
        # Breakout conditions
        long_breakout = close[i] > camarilla_h3_aligned[i]
        short_breakout = close[i] < camarilla_l3_aligned[i]
        
        # Exit conditions: return to opposite Camarilla level
        long_exit = close[i] < camarilla_l3_aligned[i]
        short_exit = close[i] > camarilla_h3_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and vol_confirm and ranging_market
        short_entry = short_breakout and vol_confirm and ranging_market
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals