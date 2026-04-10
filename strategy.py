#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and choppiness regime filter
# - Long when price breaks above 20-period Donchian high + 1w volume > 2.0x 20-period volume SMA + CHOP > 61.8 (range market)
# - Short when price breaks below 20-period Donchian low + 1w volume > 2.0x 20-period volume SMA + CHOP > 61.8 (range market)
# - Exit: price returns to opposite Donchian level
# - Position sizing: 0.25 discrete level
# - Works in bull/bear: Donchian adapts to volatility, volume spike confirms institutional interest, CHOP filter avoids trending markets where breakouts fail
# - 1d timeframe targets 7-25 trades/year with strict entry conditions

name = "1d_1w_donchian_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 20-period Donchian channels on 1d
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w volume SMA(20) for confirmation
    volume_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    # Calculate Choppiness Index on 1w (14-period)
    # True Range
    tr1 = np.maximum(df_1w['high'] - df_1w['low'], 
                     np.maximum(np.abs(df_1w['high'] - np.roll(df_1w['close'], 1)), 
                                np.abs(df_1w['low'] - np.roll(df_1w['close'], 1))))
    tr1[0] = df_1w['high'].iloc[0] - df_1w['low'].iloc[0]
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = df_1w['high'].rolling(window=14, min_periods=14).max().values
    ll = df_1w['low'].rolling(window=14, min_periods=14).min().values
    # Choppiness Index: 100 * log10(sum(tr1)/(hh-ll)) / log10(14)
    chop_raw = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
    chop = align_htf_to_ltf(prices, df_1w, chop_raw)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_sma_20_1w_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1w volume > 2.0x 20-period SMA (spike)
        vol_1w_current = align_htf_to_ltf(prices, df_1w, df_1w['volume'].values)
        vol_confirm = vol_1w_current[i] > 2.0 * volume_sma_20_1w_aligned[i]
        
        # Regime filter: CHOP > 61.8 indicates ranging market (good for mean reversion breakouts)
        ranging_market = chop[i] > 61.8
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Exit conditions: return to opposite Donchian level
        long_exit = close[i] < donchian_low[i]
        short_exit = close[i] > donchian_high[i]
        
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