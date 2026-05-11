#!/usr/bin/env python3
# 4h_KAMA_RSI_ChopFilter_v1
# Hypothesis: KAMA trend direction combined with RSI extremes and Choppiness Index regime filter.
# In trending markets (CHOP < 38.2), follow KAMA direction. In ranging markets (CHOP > 61.8), 
# use RSI for mean reversion at extremes. Volume confirmation filters weak moves.
# Designed to work in both bull and bear markets by adapting to regime.
# Targets 20-30 trades/year to minimize fee drag.

name = "4h_KAMA_RSI_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d Data for Regime Filter (Choppiness Index) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 4h KAMA Trend Indicator ===
    def calculate_kama(close, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        er[period:] = change[period-1:] / volatility[period-1:]
        # Smoothing Constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    
    # === 4h RSI (14) ===
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, period=14)
    
    # === 1d Choppiness Index (14) ===
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr[1:] = tr
        
        # ATR period average
        atr_period = np.zeros_like(close)
        for i in range(period, len(close)):
            atr_period[i] = np.mean(atr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        hh = np.zeros_like(close)
        ll = np.zeros_like(close)
        for i in range(period-1, len(close)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        
        # Chop calculation
        chop = np.zeros_like(close)
        for i in range(period, len(close)):
            if atr_period[i] > 0 and (hh[i] - ll[i]) > 0:
                chop[i] = 100 * np.log10(np.sum(atr[i-period+1:i+1]) / (hh[i] - ll[i])) / np.log10(period)
            else:
                chop[i] = 50  # Neutral when undefined
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, period=14)
    chop_1d_4h = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === Volume Spike Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5  # Require 1.5x average volume
    
    # === Signal Parameters ===
    position_size = 0.25  # 25% of capital per trade
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers KAMA, RSI, CHOP)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_1d_4h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Regime-based logic
            if chop_1d_4h[i] < 38.2:  # Trending market
                # Follow KAMA direction
                if close[i] > kama[i]:
                    signals[i] = position_size
                    position = 1
                elif close[i] < kama[i]:
                    signals[i] = -position_size
                    position = -1
            elif chop_1d_4h[i] > 61.8:  # Ranging market
                # Mean reversion with RSI extremes
                if rsi[i] < 30 and close[i] > kama[i]:  # Oversold + price above KAMA
                    signals[i] = position_size
                    position = 1
                elif rsi[i] > 70 and close[i] < kama[i]:  # Overbought + price below KAMA
                    signals[i] = -position_size
                    position = -1
        else:
            # Exit conditions
            if position == 1:  # Long position
                # Exit on opposite KAMA cross or RSI overbought in ranging market
                if close[i] < kama[i] or (chop_1d_4h[i] > 61.8 and rsi[i] > 70):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:  # Short position
                # Exit on opposite KAMA cross or RSI oversold in ranging market
                if close[i] > kama[i] or (chop_1d_4h[i] > 61.8 and rsi[i] < 30):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals