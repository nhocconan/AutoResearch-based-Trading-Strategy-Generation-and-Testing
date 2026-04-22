#!/usr/bin/env python3
"""
Hypothesis: 4h KAMA trend + RSI mean reversion with volume confirmation.
Long when KAMA turns upward, RSI < 40, and volume spike.
Short when KAMA turns downward, RSI > 60, and volume spike.
Exit when KAMA reverses or RSI reaches extreme.
Designed for low trade frequency (20-40/year) to minimize fee drift.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for KAMA trend filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily KAMA
    close_d = pd.Series(df_daily['close'].values)
    # Efficiency Ratio
    change = abs(close_d.diff(10))
    volatility = close_d.diff().abs().rolling(10).sum()
    er = change / volatility
    er = er.fillna(0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    sc = sc.fillna(0)
    # KAMA calculation
    kama = np.zeros(len(close_d))
    kama[0] = close_d.iloc[0]
    for i in range(1, len(close_d)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_d.iloc[i] - kama[i-1])
    kama = kama
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_daily, kama)
    
    # Calculate 4h RSI (14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after lookback
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA turning up, RSI oversold, volume spike
            if (i > 20 and kama_aligned[i] > kama_aligned[i-1] and 
                rsi[i] < 40 and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down, RSI overbought, volume spike
            elif (i > 20 and kama_aligned[i] < kama_aligned[i-1] and 
                  rsi[i] > 60 and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: KAMA turns down OR RSI overbought
                if kama_aligned[i] < kama_aligned[i-1] or rsi[i] > 70:
                    exit_signal = True
            else:  # position == -1
                # Exit short: KAMA turns up OR RSI oversold
                if kama_aligned[i] > kama_aligned[i-1] or rsi[i] < 30:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_KAMA_RSI_Volume_MeanReversion"
timeframe = "4h"
leverage = 1.0
#%%