#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h volume-weighted RSI with 12h trend filter and 24h ATR stoploss.
# Uses RSI(14) weighted by volume to filter false signals, combined with 12h EMA trend.
# Long when volume-RSI < 30 and price > 12h EMA; short when volume-RSI > 70 and price < 12h EMA.
# Designed to work in both bull (follow 12h uptrend) and bear (follow 12h downtrend) markets.
# Target: 20-40 trades/year to avoid fee drag.
name = "4h_VolumeRSI_12hEMA_ATRStop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate volume-weighted RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Weight gains and losses by volume
    vol_weight = volume / (np.mean(volume) + 1e-8)
    weighted_gain = gain * vol_weight
    weighted_loss = loss * vol_weight
    
    # Calculate smoothed averages
    avg_gain = pd.Series(weighted_gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(weighted_loss).ewm(alpha=1/14, adjust=False).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    vwrsi = 100 - (100 / (1 + rs))
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 14  # Need enough data for RSI and ATR
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(vwrsi[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: vwrsi < 30 (oversold) + price > 12h EMA (uptrend)
            if vwrsi[i] < 30 and price > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Enter short: vwrsi > 70 (overbought) + price < 12h EMA (downtrend)
            elif vwrsi[i] > 70 and price < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss: price < entry - 2*ATR
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit long: vwrsi > 50 (mean reversion) or trend reverse
            elif vwrsi[i] > 50 or price < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss: price > entry + 2*ATR
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit short: vwrsi < 50 (mean reversion) or trend reverse
            elif vwrsi[i] < 50 or price > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals