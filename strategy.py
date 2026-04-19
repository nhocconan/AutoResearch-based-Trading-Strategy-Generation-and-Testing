#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_RSI_ChopFilter_V1"
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
    
    # Get 1d data (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d KAMA (adaptive moving average)
    def kama(price, period=10, fast=2, slow=30):
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=1)
        er = np.zeros_like(price)
        er[period:] = change[period-1:] / (volatility[period-1:] + 1e-10)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(price)
        kama[period] = price[period]
        for i in range(period+1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    kama_1d = kama(close_1d, 10, 2, 30)
    kama_4h = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # 1d RSI
    def rsi(price, period=14):
        delta = np.diff(price)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
        avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        rsi = np.concatenate([np.full(period, np.nan), rsi])
        return rsi
    
    rsi_1d = rsi(close_1d, 14)
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1d Choppiness Index
    def choppiness_index(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=1, min_periods=1).sum().values  # sum of TR
        atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
        max_h = pd.Series(high).rolling(window=period, min_periods=period).max().values
        min_l = pd.Series(low).rolling(window=period, min_periods=period).min().values
        chop = 100 * np.log10(atr_sum / (max_h - min_l + 1e-10)) / np.log10(period)
        return chop
    
    chop_1d = choppiness_index(high_1d, low_1d, close_1d, 14)
    chop_4h = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # 4h ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_4h = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(chop_4h[i]) or \
           np.isnan(atr_4h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        kama = kama_4h[i]
        rsi_val = rsi_4h[i]
        chop = chop_4h[i]
        atr = atr_4h[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        # Chop > 61.8 = ranging (mean revert), Chop < 38.2 = trending
        chop_filter = chop > 61.8  # Only trade in ranging markets
        
        if position == 0:
            # Long: Price > KAMA + RSI < 30 (oversold) + chop filter + volume
            if price > kama and rsi_val < 30 and chop_filter and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price < KAMA + RSI > 70 (overbought) + chop filter + volume
            elif price < kama and rsi_val > 70 and chop_filter and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price < KAMA OR RSI > 70 OR ATR stop (1.5x ATR from entry)
            if price < kama or rsi_val > 70 or price < (high[i] - 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price > KAMA OR RSI < 30 OR ATR stop (1.5x ATR from entry)
            if price > kama or rsi_val < 30 or price > (low[i] + 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals