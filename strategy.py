#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA (Kaufman Adaptive Moving Average) trend with RSI(14) mean reversion and choppiness regime filter
# KAMA adapts to market noise - fast in trends, slow in ranging markets
# Long when: price > KAMA(10,2,30) AND RSI(14) < 40 (oversold pullback in uptrend) AND Chop(14) > 61.8 (ranging market)
# Short when: price < KAMA(10,2,30) AND RSI(14) > 60 (overbought bounce in downtrend) AND Chop(14) > 61.8 (ranging market)
# Uses 1w EMA(50) as higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation: volume > 1.5x 20-period EMA to ensure participation
# Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag in ranging/bear markets

name = "1d_KAMA_RSI_Chop_Volume_1wEMA50"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA, RSI, Chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate KAMA(10,2,30) on 1d close
    close_1d = df_1d['close'].values
    # Efficiency ratio
    change = np.abs(np.diff(close_1d, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=1)  # 10-period sum of abs changes
    # Pad the beginning with NaN for alignment
    change_padded = np.concatenate([np.full(10, np.nan), change])
    volatility_padded = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility_padded != 0, change_padded / volatility_padded, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start with first close
    for i in range(10, len(close_1d)):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = close_1d[i]
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14) on 1d close
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Pad the beginning
    gain_padded = np.concatenate([np.array([np.nan]), gain])
    loss_padded = np.concatenate([np.array([np.nan]), loss])
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    # Wilder's smoothing
    avg_gain[13] = np.nanmean(gain_padded[1:15])  # First 14 gains
    avg_loss[13] = np.nanmean(loss_padded[1:15])  # First 14 losses
    
    for i in range(14, len(close_1d)):
        if not np.isnan(avg_gain[i-1]) and not np.isnan(avg_loss[i-1]):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain_padded[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss_padded[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate Choppiness Index(14) on 1d high/low/close
    # True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr_padded = np.concatenate([np.array([np.nan]), tr])
    
    # ATR(14) - sum of TR over 14 periods
    atr_14 = np.full_like(close_1d, np.nan)
    for i in range(14, len(close_1d)):
        atr_14[i] = np.nansum(tr_padded[i-13:i+1])  # 14-period sum
    
    # Highest high and lowest low over 14 periods
    hh_14 = np.full_like(close_1d, np.nan)
    ll_14 = np.full_like(close_1d, np.nan)
    for i in range(13, len(close_1d)):
        hh_14[i] = np.nanmax(high[i-12:i+1])
        ll_14[i] = np.nanmin(low[i-12:i+1])
    
    # Chop = 100 * log10(sum(ATR) / (HH - LL)) / log10(14)
    hh_ll = hh_14 - ll_14
    chop = np.full_like(close_1d, np.nan)
    mask = (hh_ll > 0) & (~np.isnan(atr_14)) & (~np.isnan(hh_ll))
    chop[mask] = 100 * np.log10(atr_14[mask] / hh_ll[mask]) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: 20-period EMA on 1d volume
    vol_1d = df_1d['volume'].values
    vol_ema_20 = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA (moderate to avoid overtrading)
        volume_spike = volume[i] > (1.5 * vol_ema_20_aligned[i])
        
        # Chop regime: ranging market (Chop > 61.8)
        chop_regime = chop_aligned[i] > 61.8
        
        # KAMA + RSI signals with 1w trend filter and volume confirmation
        if position == 0:
            if (close[i] > kama_aligned[i] and  # Price above KAMA (uptrend bias)
                rsi_aligned[i] < 40 and        # RSI oversold
                chop_regime and                # Ranging market
                volume_spike and               # Volume confirmation
                close[i] > ema_50_1w_aligned[i]):  # Price above 1w EMA50 (uptrend)
                signals[i] = 0.25
                position = 1
            elif (close[i] < kama_aligned[i] and   # Price below KAMA (downtrend bias)
                  rsi_aligned[i] > 60 and          # RSI overbought
                  chop_regime and                  # Ranging market
                  volume_spike and                 # Volume confirmation
                  close[i] < ema_50_1w_aligned[i]):  # Price below 1w EMA50 (downtrend)
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA OR RSI > 50 (momentum fade)
            if close[i] < kama_aligned[i] or rsi_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA OR RSI < 50 (momentum fade)
            if close[i] > kama_aligned[i] or rsi_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals