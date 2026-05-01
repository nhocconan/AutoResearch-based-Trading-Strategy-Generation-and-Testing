#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI(14) + volume confirmation + chop regime filter
# Long when KAMA direction is up (price > KAMA) AND RSI < 30 (oversold) AND volume > 1.5x 20d median AND chop > 61.8 (range)
# Short when KAMA direction is down (price < KAMA) AND RSI > 70 (overbought) AND volume > 1.5x 20d median AND chop > 61.8 (range)
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# KAMA adapts to market noise, RSI captures mean reversion in range, volume confirms interest, chop ensures ranging market.
# Target: 10-20 trades/year on 1d timeframe (40-80 total over 4 years) to minimize fee drag.
# This combination has shown strong test performance in DB for SOL with proper filtering and should work on BTC/ETH.

name = "1d_KAMA_Direction_RSI_Volume_Chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros_like(close)
    er[10:] = change[10:] / volatility[10:]
    er[volatility == 0] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Handle first 14 values
    rsi[:14] = 50  # neutral
    
    # Calculate 20-day volume median for confirmation
    vol_median_20d = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Choppiness Index (CHOP) over 14 periods
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Sum of TR over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # CHOP = 100 * log10(atr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    hh_ll = hh - ll
    chop = np.zeros_like(close)
    mask = (hh_ll > 0) & (atr_sum > 0)
    chop[mask] = 100 * np.log10(atr_sum[mask] / hh_ll[mask]) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for KAMA, RSI, ATR, volume, chop
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_median_20d[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-day volume median
        if vol_median_20d[i] <= 0 or np.isnan(vol_median_20d[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20d[i] * 1.5)
        
        # Chop regime filter: chop > 61.8 indicates ranging market (mean reversion favorable)
        chop_filter = chop[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long: KAMA up (price > KAMA) AND RSI < 30 (oversold) AND volume confirm AND chop filter
            if (curr_close > kama[i] and 
                rsi[i] < 30 and 
                volume_confirm and 
                chop_filter):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: KAMA down (price < KAMA) AND RSI > 70 (overbought) AND volume confirm AND chop filter
            elif (curr_close < kama[i] and 
                  rsi[i] > 70 and 
                  volume_confirm and 
                  chop_filter):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price < KAMA (trend change) OR RSI > 50 (mean reversion exhausted)
            elif (curr_close < kama[i]) or (rsi[i] > 50):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price > KAMA (trend change) OR RSI < 50 (mean reversion exhausted)
            elif (curr_close > kama[i]) or (rsi[i] < 50):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals