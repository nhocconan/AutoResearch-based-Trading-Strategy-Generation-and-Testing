#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA direction with 1d RSI filter and volume spike
# Long when KAMA is rising (bullish trend) and RSI < 40 (pullback in uptrend) with volume spike
# Short when KAMA is falling (bearish trend) and RSI > 60 (bounce in downtrend) with volume spike
# Exit when KAMA direction changes or RSI reaches extreme levels (70/30)
# Designed for low trade frequency (~15-30/year) on 12h timeframe to minimize fee drain.
# KAMA adapts to market efficiency, reducing whipsaw in choppy markets.
# Works in bull/bear by trading pullbacks in the direction of the trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for RSI and volume context
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 14-period RSI on 1d close
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values  # fill NaN with 50 for periods < 14
    
    # Calculate KAMA on 12h close
    close = prices['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    # Handle edge cases for first 10 values
    er = np.full_like(change, np.nan, dtype=float)
    er[10:] = change[10:] / volatility[10:]
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # seed
    for i in range(10, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align 1d RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        kama_val = kama[i]
        kama_prev = kama[i-1]
        rsi_val = rsi_aligned[i]
        
        # Volume filter: current volume > 1.8 * 20-period average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: KAMA rising (bullish) + RSI < 40 (pullback) + volume spike
            if kama_val > kama_prev and rsi_val < 40 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA falling (bearish) + RSI > 60 (bounce) + volume spike
            elif kama_val < kama_prev and rsi_val > 60 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: KAMA direction changes or RSI reaches extreme
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when KAMA turns down or RSI > 70 (overbought)
                if kama_val < kama_prev or rsi_val > 70:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when KAMA turns up or RSI < 30 (oversold)
                if kama_val > kama_prev or rsi_val < 30:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_KAMA_RSI_Volume"
timeframe = "12h"
leverage = 1.0