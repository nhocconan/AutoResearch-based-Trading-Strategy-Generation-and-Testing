#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ATR(14) for volatility normalization
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d Donchian Channel (20-period)
    dc_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    dc_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    dc_middle = (dc_upper + dc_lower) / 2
    
    # 1d RSI(14) for momentum
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean().values
    avg_loss = loss.rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d Volume spike detection
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 2.0)
    
    # Align HTF indicators to 6h timeframe
    dc_upper_aligned = align_htf_to_ltf(prices, df_1d, dc_upper)
    dc_lower_aligned = align_htf_to_ltf(prices, df_1d, dc_lower)
    dc_middle_aligned = align_htf_to_ltf(prices, df_1d, dc_middle)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(dc_upper_aligned[i]) or np.isnan(dc_lower_aligned[i]) or 
            np.isnan(dc_middle_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > dc_upper_aligned[i]
        breakout_down = close[i] < dc_lower_aligned[i]
        
        # Momentum filter: RSI in favorable range (avoid chop)
        rsi_momentum_up = rsi_aligned[i] > 55  # Bullish momentum
        rsi_momentum_down = rsi_aligned[i] < 45  # Bearish momentum
        
        # Volatility filter: require volatility to be elevated
        vol_cond = vol_spike_aligned[i]
        
        # Entry conditions - Donchian breakout with momentum and volume
        long_entry = breakout_up and rsi_momentum_up and vol_cond
        short_entry = breakout_down and rsi_momentum_down and vol_cond
        
        # Exit conditions: return to middle Donchian or opposite breakout
        long_exit = close[i] < dc_middle_aligned[i] or breakout_down
        short_exit = close[i] > dc_middle_aligned[i] or breakout_up
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_Breakout_RSI_Volume"
timeframe = "6h"
leverage = 1.0