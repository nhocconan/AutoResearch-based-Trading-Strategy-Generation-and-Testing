#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Chandelier_Exit_With_Volume_Spike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # === Weekly: Chandelier Exit components ===
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # ATR(22) for weekly (approx 5 months)
    tr1_w = np.abs(high_w[1:] - low_w[1:])
    tr2_w = np.abs(high_w[1:] - close_w[:-1])
    tr3_w = np.abs(low_w[1:] - close_w[:-1])
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    tr_w = np.concatenate([[np.nan], tr_w])
    atr_w = pd.Series(tr_w).rolling(window=22, min_periods=22).mean().values
    
    # 22-period high and low for Chandelier Exit
    high_max_w = pd.Series(high_w).rolling(window=22, min_periods=22).max().values
    low_min_w = pd.Series(low_w).rolling(window=22, min_periods=22).min().values
    
    # Chandelier Exit: Long exit = 22-period high - 3*ATR, Short exit = 22-period low + 3*ATR
    chandelier_long_exit = high_max_w - 3.0 * atr_w
    chandelier_short_exit = low_min_w + 3.0 * atr_w
    
    # Align Chandelier levels to daily
    chandelier_long_exit_aligned = align_htf_to_ltf(prices, df_weekly, chandelier_long_exit)
    chandelier_short_exit_aligned = align_htf_to_ltf(prices, df_weekly, chandelier_short_exit)
    
    # === Daily: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume condition: current volume > 2.0x 20-day average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        long_exit = chandelier_long_exit_aligned[i]
        short_exit = chandelier_short_exit_aligned[i]
        current_close = close[i]
        current_volume = volume[i]
        current_vol_ma = vol_ma_20[i]
        
        # Skip if any value is NaN
        if (np.isnan(long_exit) or np.isnan(short_exit) or np.isnan(current_vol_ma)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0x 20-day average
        vol_condition = current_volume > 2.0 * current_vol_ma
        
        if position == 0:
            # Long: price breaks above Chandelier long exit with volume confirmation
            if current_close > long_exit and vol_condition:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short: price breaks below Chandelier short exit with volume confirmation
            elif current_close < short_exit and vol_condition:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: price closes below Chandelier long exit
            if current_close < long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above Chandelier short exit
            if current_close > short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals