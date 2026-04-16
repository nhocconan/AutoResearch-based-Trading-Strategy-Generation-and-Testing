#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 4h RSI mean reversion + 1d volume confirmation
# In high chop (range) markets: RSI < 30 = long, RSI > 70 = short
# In low chop (trending) markets: avoid trades to prevent whipsaw
# Volume confirmation ensures institutional participation
# Target: 50-120 total trades over 4 years (12-30/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h RSI(14) for mean reversion ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 4h Choppiness Index(14) for regime detection ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low + 1e-10)) / np.log10(14)
    
    # === 1d Volume Confirmation (using 4h data: 6 periods = 1 day) ===
    vol_ma_1d = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 30
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or
            np.isnan(vol_ma_1d[i])):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        chop_val = chop[i]
        vol_confirm = volume[i] > vol_ma_1d[i] * 1.5  # 1.5x average volume
        
        # Only trade in high chop (range) markets: CHOP > 61.8
        if chop_val > 61.8:
            # Mean reversion: RSI < 30 = long, RSI > 70 = short
            if rsi_val < 30 and vol_confirm:
                signals[i] = 0.25
            elif rsi_val > 70 and vol_confirm:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:
            # Low chop (trending) market: avoid trades
            signals[i] = 0.0
    
    return signals

name = "4h_Chop61.8_RSI30_70_Volume1.5x"
timeframe = "4h"
leverage = 1.0