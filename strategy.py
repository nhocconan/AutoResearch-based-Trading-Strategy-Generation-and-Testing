#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h KAMA trend with 1d RSI mean reversion and volume confirmation.
# Long when 12h KAMA is rising AND 1d RSI < 30 AND volume > 1.5x 20-period average.
# Short when 12h KAMA is falling AND 1d RSI > 70 AND volume > 1.5x 20-period average.
# Exit when RSI crosses 50 or KAMA trend reverses.
# Combines trend following (KAMA) with mean reversion (RSI) to capture swings in both bull and bear markets.
# Volume ensures participation, reducing false signals. Target: 25-40 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for KAMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate KAMA (10-period)
    def kama(close, length=10):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
        er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
        # Smoothing constants
        sc = np.power(er * (2/(2+1) - 2/(30+1)) + 2/(30+1), 2)
        # KAMA calculation
        kama_out = np.full_like(close, np.nan)
        kama_out[length-1] = close[length-1]
        for i in range(length, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i-length] * (close[i] - kama_out[i-1])
        return kama_out
    
    kama_12h = kama(close_12h, 10)
    kama_rising = np.zeros_like(kama_12h, dtype=bool)
    kama_rising[1:] = kama_12h[1:] > kama_12h[:-1]
    
    # Load 1d data ONCE for RSI and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate RSI (14-period)
    def rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        for i in range(length+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi_out = 100 - (100 / (1 + rs))
        return rsi_out
    
    rsi_1d = rsi(close_1d, 14)
    
    # Calculate 20-period average volume
    vol_ma_20 = np.full_like(volume_1d, np.nan)
    for i in range(19, len(volume_1d)):
        vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align indicators to 4h timeframe
    kama_rising_aligned = align_htf_to_ltf(prices, df_12h, kama_rising)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(kama_rising_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current daily volume vs 20-day average
        volume_ratio = volume_1d_aligned[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        if position == 0:
            # Look for mean reversion entries with trend and volume confirmation
            # Long: KAMA rising AND RSI < 30 AND volume > 1.5x average
            if (kama_rising_aligned[i] and 
                rsi_1d_aligned[i] < 30 and 
                volume_ratio > 1.5):
                position = 1
                signals[i] = position_size
            # Short: KAMA falling AND RSI > 70 AND volume > 1.5x average
            elif ((not kama_rising_aligned[i]) and 
                  rsi_1d_aligned[i] > 70 and 
                  volume_ratio > 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses 50 or KAMA trend reverses
            if (rsi_1d_aligned[i] > 50 or 
                not kama_rising_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses 50 or KAMA trend reverses
            if (rsi_1d_aligned[i] < 50 or 
                kama_rising_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_KAMA_1d_RSI_Volume_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0