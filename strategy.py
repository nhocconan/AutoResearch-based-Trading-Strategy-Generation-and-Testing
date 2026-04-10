#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band breakout with 1d volume confirmation and chop regime filter
# - Long when price closes above upper BB(20,2) + Bollinger Band Width < 0.06 (squeeze) + 1d volume > 1.4x 20-period volume SMA + Chop(14) > 55
# - Short when price closes below lower BB(20,2) + Bollinger Band Width < 0.06 (squeeze) + 1d volume > 1.4x 20-period volume SMA + Chop(14) > 55
# - Exit: price crosses back through middle BB(20) line
# - Position sizing: 0.25 discrete level
# - Bollinger Band squeeze identifies low volatility periods primed for breakout
# - Volume confirms breakout validity, chop filter avoids weak trends
# - 4h timeframe targets 20-50 trades/year with strict entry conditions to minimize fee drag

name = "4h_1d_bb_squeeze_volume_chop_v2"
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
    
    # Calculate 4h Bollinger Bands (20,2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma + (bb_std * std)
    lower_bb = sma - (bb_std * std)
    middle_bb = sma
    
    # Calculate Bollinger Band Width for squeeze detection
    bb_width = (upper_bb - lower_bb) / middle_bb
    bb_width = np.where(np.isnan(bb_width), 0, bb_width)
    
    # Calculate 4h Chopiness Index (14-period) for regime filter
    # True Range
    tr1 = np.maximum(high - low, 
                     np.maximum(np.abs(high - np.roll(close, 1)), 
                                np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    # Sum of TR over period
    sum_tr = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over period
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Chop formula: 100 * log10(sum_TR / (HH - LL)) / log10(N)
    # Avoid division by zero and log of zero/negative
    hl_range = hh - ll
    chop = np.where((hl_range > 0) & (sum_tr > 0), 
                    100 * np.log10(sum_tr / hl_range) / np.log10(14), 
                    50)  # default to neutral when invalid
    
    # Calculate 1d OHLC for volume confirmation
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume SMA(20) for confirmation
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(sma[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(bb_width[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for volume spike confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Volume confirmation: current 1d volume > 1.4x 20-period SMA (volume spike)
        vol_confirm = vol_1d_current[i] > 1.4 * volume_sma_20_1d_aligned[i]
        
        # Squeeze condition: Bollinger Band Width < 0.06 indicates low volatility
        squeeze_condition = bb_width[i] < 0.06
        
        # Regime filter: Chop > 55 indicates ranging/transition market (favorable for breakout)
        favorable_regime = chop[i] > 55
        
        # Bollinger Band signals
        long_entry = (close[i] > upper_bb[i]) and squeeze_condition and vol_confirm and favorable_regime
        short_entry = (close[i] < lower_bb[i]) and squeeze_condition and vol_confirm and favorable_regime
        exit_long = close[i] < middle_bb[i]  # Exit long when price crosses below middle BB
        exit_short = close[i] > middle_bb[i]  # Exit short when price crosses above middle BB
        
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
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals