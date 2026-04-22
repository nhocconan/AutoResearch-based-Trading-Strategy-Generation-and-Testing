#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI(14) with 1d EMA50 trend filter and volume spike confirmation.
# RSI < 30 indicates oversold conditions, RSI > 70 indicates overbought conditions.
# Combined with 1d EMA50 trend filter to ensure we trade in the direction of higher timeframe trend,
# and volume spikes (>1.5x 20-period average) to confirm institutional participation.
# Designed for low trade frequency (~15-25/year) to minimize fee decay.
# Works in both bull and bear markets by following higher timeframe trend and buying dips/selling rallies.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for RSI calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI on 1d close
    delta = pd.Series(close_1d).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate 50-period EMA on 1d close for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 1h timeframe (waits for 1d bar to close)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        rsi_val = rsi_aligned[i]
        ema_val = ema_50_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average (moderate filter for low frequency)
        vol_spike = vol > 1.5 * vol_ma
        
        # RSI conditions
        rsi_oversold = rsi_val < 30
        rsi_overbought = rsi_val > 70
        
        if position == 0:
            # Long conditions: RSI oversold + price above EMA + volume spike
            if rsi_oversold and price > ema_val and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short conditions: RSI overbought + price below EMA + volume spike
            elif rsi_overbought and price < ema_val and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when RSI becomes overbought or price breaks below EMA
                if rsi_overbought or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when RSI becomes oversold or price breaks above EMA
                if rsi_oversold or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1d_RSI14_1dEMA50_Volume"
timeframe = "1h"
leverage = 1.0