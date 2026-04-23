#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Williams %R extremes with volume confirmation and ATR trailing stop.
Long when 1d Williams %R crosses above -80 from below AND volume > 1.5x 20-period average.
Short when 1d Williams %R crosses below -20 from above AND volume > 1.5x 20-period average.
Exit when price retraces to 50% of the move from extreme OR ATR trailing stop (2.5*ATR) hit.
Uses discrete position sizing (0.30) to balance return and drawdown.
Williams %R identifies overbought/oversold conditions on higher timeframe, volume confirms momentum,
and ATR trailing stop manages risk in both trending and ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    williams_r_high = -100  # track highest Williams %R during long
    williams_r_low = 0      # track lowest Williams %R during short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        wr = williams_r_aligned[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND volume spike
            if i > start_idx:
                prev_wr = williams_r_aligned[i-1]
                if (prev_wr <= -80 and wr > -80 and volume[i] > 1.5 * vol_ma_val):
                    signals[i] = 0.30
                    position = 1
                    entry_price = price
                    williams_r_high = wr
            # Short: Williams %R crosses below -20 from above AND volume spike
            elif i > start_idx:
                prev_wr = williams_r_aligned[i-1]
                if (prev_wr >= -20 and wr < -20 and volume[i] > 1.5 * vol_ma_val):
                    signals[i] = -0.30
                    position = -1
                    entry_price = price
                    williams_r_low = wr
        else:
            # Update extreme Williams %R during position
            if position == 1:
                williams_r_high = max(williams_r_high, wr)
            elif position == -1:
                williams_r_low = min(williams_r_low, wr)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to 50% of the move from extreme
            if position == 1 and williams_r_high > -100:
                # Normalize Williams %R to 0-100 scale for retracement calculation
                wr_norm = (wr + 100)  # 0 to 100
                wr_high_norm = (williams_r_high + 100)  # 0 to 100
                # Exit when Williams %R retraces 50% from extreme
                if wr_norm <= wr_high_norm * 0.5:
                    exit_signal = True
            elif position == -1 and williams_r_low < 0:
                # Normalize Williams %R to 0-100 scale for retracement calculation
                wr_norm = (wr + 100)  # 0 to 100
                wr_low_norm = (williams_r_low + 100)  # 0 to 100
                # Exit when Williams %R retraces 50% from extreme (toward 100)
                if wr_norm >= wr_low_norm + (100 - wr_low_norm) * 0.5:
                    exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from extreme price
            if position == 1:
                # Track highest price during long for trailing stop
                if not hasattr(generate_signals, 'long_high_price'):
                    generate_signals.long_high_price = entry_price
                generate_signals.long_high_price = max(generate_signals.long_high_price, price)
                if price < generate_signals.long_high_price - 2.5 * atr_val:
                    exit_signal = True
            elif position == -1:
                # Track lowest price during short for trailing stop
                if not hasattr(generate_signals, 'short_low_price'):
                    generate_signals.short_low_price = entry_price
                generate_signals.short_low_price = min(generate_signals.short_low_price, price)
                if price > generate_signals.short_low_price + 2.5 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                williams_r_high = -100
                williams_r_low = 0
                # Reset tracking variables
                if hasattr(generate_signals, 'long_high_price'):
                    delattr(generate_signals, 'long_high_price')
                if hasattr(generate_signals, 'short_low_price'):
                    delattr(generate_signals, 'short_low_price')
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_WilliamsR_VolumeConfirmation_50PercentRetracement_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0