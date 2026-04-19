#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_RSI_ChopFilter_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # KAMA on close (ER=10, fast=2, slow=30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0]).reshape(-1, 10)), axis=1)
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    sc = np.power(er * (2/2 - 2/30) + 2/30, 2)
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama = np.where(np.arange(len(close)) < 10, np.nan, kama)
    
    # Daily ATR for volatility filter
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr_14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness index (14)
    atr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    
    # Volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        atr = atr_14[i]
        vol_ma = vol_ma_20[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        chop_filter = chop_val > 50  # avoid strong trends
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, chop filter, volume
            if price > kama_val and rsi_val > 50 and chop_filter and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50, chop filter, volume
            elif price < kama_val and rsi_val < 50 and chop_filter and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below KAMA or RSI < 40
            if price < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above KAMA or RSI > 60
            if price > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals