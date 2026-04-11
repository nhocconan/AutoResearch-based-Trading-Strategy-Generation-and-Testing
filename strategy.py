#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prrices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Calculate weekly ATR for volatility filter
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = np.abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly high/low for breakout levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Align weekly levels to 12h timeframe
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Volume confirmation: 12h volume > 2.0x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(high_1w_aligned[i]) or np.isnan(low_1w_aligned[i]) or
            np.isnan(atr_1w_aligned[i]) or np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 2.0 * vol_ma_50[i]
        
        # Breakout conditions using weekly high/low
        breakout_up = price_close > high_1w_aligned[i]  # Break above weekly high
        breakout_down = price_close < low_1w_aligned[i]  # Break below weekly low
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Break above weekly high with volume confirmation
        if breakout_up and vol_confirm:
            enter_long = True
        
        # Short: Break below weekly low with volume confirmation
        if breakout_down and vol_confirm:
            enter_short = True
        
        # Exit conditions: return to opposite weekly level
        exit_long = price_close < low_1w_aligned[i]  # Return to weekly low
        exit_short = price_close > high_1w_aligned[i]  # Return to weekly high
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.30
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 12h weekly breakout strategy using actual weekly high/low levels with volume confirmation.
# Enters long when price breaks above weekly high with volume > 2.0x 50-period average.
# Enters short when price breaks below weekly low with volume > 2.0x 50-period average.
# Exits when price returns to the opposite weekly level.
# Uses higher volume threshold (2.0x) and longer MA (50) to reduce trade frequency.
# Position size set to 0.30 to capture trend moves while managing risk.
# Target: 15-25 trades per year (60-100 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by capturing significant breakouts in either direction.
# 12h timeframe provides good balance between signal quality and trade frequency.
# Weekly timeframe filter ensures we only trade significant breakouts.