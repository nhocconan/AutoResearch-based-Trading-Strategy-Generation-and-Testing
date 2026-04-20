#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day KAMA + RSI + chop regime (mean reversion in chop, trend follow in trend)
# - Uses daily KAMA for trend direction: price above KAMA = long bias, below = short bias
# - Entry: RSI reverses from extreme (long when RSI<35, short when RSI>65) with volume confirmation
# - Filter: Choppiness index > 61.8 for mean reversion mode, < 38.2 for trend mode
# - Exit: Opposite RSI extreme or trend change (price crosses KAMA)
# - Target: 15-25 trades per year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate KAMA on 1d data
    # Efficiency ratio: |close - close(10)| / sum|close - close(1)| over 10 periods
    change = np.abs(close_1d[10:] - close_1d[:-10])
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # will fix below
    # Proper ER calculation
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        price_change = np.abs(close_1d[i] - close_1d[i-10])
        price_volatility = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
        if price_volatility > 0:
            er[i] = price_change / price_volatility
        else:
            er[i] = 0
    er[:10] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_1d = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI on 1d data
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    
    # First average
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    # Wilder smoothing
    for i in range(15, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = 50  # neutral before enough data
    
    rsi_1d = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate Choppiness Index on 1d data
    # ATR(14)
    tr1 = high_1d[1:] - low_1d[:-1]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high_1d[0] - low_1d[0]], tr])
    
    atr = np.zeros_like(close_1d)
    for i in range(1, len(atr)):
        if i < 14:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR over 14 periods
    sum_atr = np.zeros_like(close_1d)
    for i in range(13, len(close_1d)):
        sum_atr[i] = np.sum(tr[i-12:i+1])
    
    # Highest high and lowest low over 14 periods
    highest_high = np.zeros_like(close_1d)
    lowest_low = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 14:
            highest_high[i] = np.max(high_1d[:i+1])
            lowest_low[i] = np.min(low_1d[:i+1])
        else:
            highest_high[i] = np.max(high_1d[i-13:i+1])
            lowest_low[i] = np.min(low_1d[i-13:i+1])
    
    # Chop = 100 * log10(sum(ATR14) / (HH - LL)) / log10(14)
    chop = np.zeros_like(close_1d)
    for i in range(13, len(close_1d)):
        if highest_high[i] > lowest_low[i]:
            chop[i] = 100 * np.log10(sum_atr[i] / (highest_high[i] - lowest_low[i])) / np.log10(14)
        else:
            chop[i] = 50
    
    chop_1d = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: 20-period average on 1d
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.zeros_like(volume_1d)
    for i in range(len(volume_1d)):
        if i < 20:
            vol_ma_1d[i] = np.mean(volume_1d[:i+1]) if i > 0 else volume_1d[i]
        else:
            vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1d price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(kama_1d[i]) or np.isnan(rsi_1d[i]) or np.isnan(chop_1d[i]) or np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_1d_aligned[i]
        chop_val = chop_1d[i]
        
        if position == 0:
            # Determine market regime
            is_chop = chop_val > 61.8  # mean reversion regime
            is_trend = chop_val < 38.2  # trend following regime
            
            if is_chop:
                # Mean reversion in chop: buy oversold, sell overbought
                if price > kama_1d[i] and rsi_1d[i] < 35 and vol > 1.5 * vol_ma:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif price < kama_1d[i] and rsi_1d[i] > 65 and vol > 1.5 * vol_ma:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            elif is_trend:
                # Trend following: buy pullbacks in uptrend, sell rallies in downtrend
                if price > kama_1d[i] and rsi_1d[i] < 40 and vol > 1.2 * vol_ma:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif price < kama_1d[i] and rsi_1d[i] > 60 and vol > 1.2 * vol_ma:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Long exit: RSI overbought OR price crosses below KAMA
            if rsi_1d[i] > 65 or price < kama_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold OR price crosses above KAMA
            if rsi_1d[i] < 35 or price > kama_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopRegime"
timeframe = "1d"
leverage = 1.0