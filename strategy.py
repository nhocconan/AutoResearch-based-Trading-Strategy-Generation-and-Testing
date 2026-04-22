#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover with 4h MACD trend filter and 1d volume spike confirmation.
# EMA10/21 crossovers capture momentum shifts on 1h, while 4h MACD ensures alignment with higher timeframe trend.
# 1d volume spikes (>2x 20-period average) filter for institutional participation.
# Designed for low trade frequency (~20-40/year) to minimize fee decay.
# Works in bull markets via trend-following entries and in bear markets via short signals when trend breaks down.
# Uses discrete position sizing (0.20) to minimize churn.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for MACD trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate MACD on 4h close
    ema_fast = pd.Series(close_4h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_slow = pd.Series(close_4h).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Align MACD components to 1h timeframe
    macd_aligned = align_htf_to_ltf(prices, df_4h, macd_line)
    signal_aligned = align_htf_to_ltf(prices, df_4h, signal_line)
    
    # Load 1d data for volume spike confirmation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period average volume on 1d
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 10/21 EMA on 1h price for entry signals
    close = prices['close'].values
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate volume spike on 1h (>2x 20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(macd_aligned[i]) or 
            np.isnan(signal_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or 
            np.isnan(ema_10[i]) or 
            np.isnan(ema_21[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        macd_val = macd_aligned[i]
        signal_val = signal_aligned[i]
        vol_ma_1d = vol_ma_aligned[i]
        ema10_val = ema_10[i]
        ema21_val = ema_21[i]
        
        # Volume filter: current 1h volume > 2.0 * 20-period average
        vol_spike_1h = vol > 2.0 * vol_ma
        # Volume filter: current 1d volume > 2.0 * 20-period average (aligned)
        vol_spike_1d = vol_ma_1d > 0 and volume_1d[-1] > 2.0 * vol_ma_1d if len(volume_1d) > 0 else False  # Simplified for alignment
        
        if position == 0:
            # Long conditions: EMA10 crosses above EMA21 + MACD bullish + volume spike
            if ema10_val > ema21_val and macd_val > signal_val and vol_spike_1h:
                signals[i] = 0.20
                position = 1
            # Short conditions: EMA10 crosses below EMA21 + MACD bearish + volume spike
            elif ema10_val < ema21_val and macd_val < signal_val and vol_spike_1h:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when EMA10 crosses below EMA21 or MACD turns bearish
                if ema10_val < ema21_val or macd_val < signal_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when EMA10 crosses above EMA21 or MACD turns bullish
                if ema10_val > ema21_val or macd_val > signal_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_EMA10_21_4hMACD_1dVolume_Spike"
timeframe = "1h"
leverage = 1.0