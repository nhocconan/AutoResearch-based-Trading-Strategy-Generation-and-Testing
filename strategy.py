#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily KAMA (20, 2, 30) - adaptive trend indicator
    def calculate_kama(close_series, er_len=10, fast_sc=2, slow_sc=30):
        change = abs(close_series - close_series.shift(er_len))
        vol = abs(close_series.diff()).rolling(window=er_len, min_periods=1).sum()
        er = np.where(vol != 0, change / vol, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.zeros_like(close_series, dtype=float)
        kama[0] = close_series.iloc[0] if hasattr(close_series, 'iloc') else close_series[0]
        for i in range(1, len(close_series)):
            kama[i] = kama[i-1] + sc[i] * (close_series[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(pd.Series(close_1d), 10, 2, 30)
    kama_dir = np.where(kama > np.roll(kama, 1), 1, -1)  # 1: up, -1: down
    
    # Daily RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily volume spike
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    vol_spike = volume_1d > (vol_ma_20 * 2.0)
    
    # Align daily indicators to 15m timeframe
    kama_dir_aligned = align_htf_to_ltf(prices, df_1d, kama_dir)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi.values)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_dir_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: KAMA direction + RSI momentum + volume spike
        long_entry = (kama_dir_aligned[i] == 1 and 
                     rsi_aligned[i] > 55 and 
                     vol_spike_aligned[i])
        short_entry = (kama_dir_aligned[i] == -1 and 
                      rsi_aligned[i] < 45 and 
                      vol_spike_aligned[i])
        
        # Exit conditions: opposite KAMA direction or RSI reversal
        long_exit = (kama_dir_aligned[i] == -1 or rsi_aligned[i] < 40)
        short_exit = (kama_dir_aligned[i] == 1 or rsi_aligned[i] > 60)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25
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

name = "1d_KAMA_RSI_Volume_Spike"
timeframe = "1d"
leverage = 1.0