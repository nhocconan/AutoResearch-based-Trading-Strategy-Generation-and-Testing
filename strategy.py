#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_KAMA_RSI_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # === Daily Indicators ===
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # KAMA on daily close
    close_series_1d = pd.Series(close_1d)
    change = abs(close_series_1d.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change / np.where(volatility > 0, volatility, 1e-10)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI(14) on daily
    delta = close_series_1d.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Chopiness Index(14) on daily
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr / (highest_high - lowest_low)) / np.log10(14)
    
    # Align daily indicators to 12h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop.values)
    
    # === 12h: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        kama_val = kama_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        chop_val = chop_aligned[i]
        vol_ratio_val = vol_ratio[i]
        close_val = close[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_val) or np.isnan(rsi_val) or 
            np.isnan(chop_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA, RSI not overbought, chop > 50 (not strong trend), volume confirmation
            if (close_val > kama_val and 
                rsi_val < 70 and 
                chop_val > 50 and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA, RSI not oversold, chop > 50, volume confirmation
            elif (close_val < kama_val and 
                  rsi_val > 30 and 
                  chop_val > 50 and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below KAMA or chop drops (trending) or volume dries up
            if close_val < kama_val or chop_val < 30 or vol_ratio_val < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above KAMA or chop drops or volume dries up
            if close_val > kama_val or chop_val < 30 or vol_ratio_val < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals