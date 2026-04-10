#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend direction + 1d RSI(14) mean reversion + volume confirmation
# - Long when 12h KAMA is rising (bullish trend) AND 1d RSI < 30 (oversold) AND 1d volume > 1.5x 20-period volume SMA
# - Short when 12h KAMA is falling (bearish trend) AND 1d RSI > 70 (overbought) AND 1d volume > 1.5x 20-period volume SMA
# - Exit: opposite RSI extreme (RSI > 50 for long exit, RSI < 50 for short exit) or KAMA direction reversal
# - Position sizing: 0.25 discrete level
# - KAMA adapts to market noise, reducing false signals in choppy markets
# - 1d RSI provides mean-reversion edge within the larger 12h trend
# - Volume confirmation ensures institutional participation
# - Target: 12-30 trades/year on 12h timeframe to minimize fee drag

name = "12h_kama_rsi_volume_v1"
timeframe = "12h"
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
    
    # Calculate 12h KAMA (adaptive trend indicator)
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=1))
    change = np.insert(change, 0, 0)  # align length
    volatility = np.abs(np.diff(close, n=1))
    volatility = np.insert(volatility, 0, 0)
    
    # Sum over 10-period window
    er_window = 10
    sum_change = pd.Series(change).rolling(window=er_window, min_periods=er_window).sum().values
    sum_volatility = pd.Series(volatility).rolling(window=er_window, min_periods=er_window).sum().values
    
    # Avoid division by zero
    er = np.where(sum_volatility == 0, 0, sum_change / sum_volatility)
    
    # Smoothing constants
    fastest = 2 / (2 + 1)   # EMA(2)
    slowest = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[er_window] = close[er_window]  # seed value
    
    for i in range(er_window + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA direction: 1 if rising, -1 if falling, 0 if flat
    kama_dir = np.zeros_like(kama)
    kama_dir[1:] = np.where(kama[1:] > kama[:-1], 1, np.where(kama[1:] < kama[:-1], -1, 0))
    
    # Calculate 1d RSI(14)
    rsi_period = 14
    delta = np.diff(df_1d['close'].values)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    
    # Avoid division by zero
    rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h timeframe
    kama_dir_aligned = align_htf_to_ltf(prices, df_1d, kama_dir)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_dir_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period SMA
        vol_confirm = volume_1d_aligned[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Entry conditions
        long_entry = (kama_dir_aligned[i] == 1) and (rsi_aligned[i] < 30) and vol_confirm
        short_entry = (kama_dir_aligned[i] == -1) and (rsi_aligned[i] > 70) and vol_confirm
        
        # Exit conditions: RSI mean reversion or trend change
        exit_long = (rsi_aligned[i] > 50) or (kama_dir_aligned[i] != 1)
        exit_short = (rsi_aligned[i] < 50) or (kama_dir_aligned[i] != -1)
        
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