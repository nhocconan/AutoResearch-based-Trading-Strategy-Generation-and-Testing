#!/usr/bin/env python3
"""
Hypothesis: 12h KAMA trend + RSI mean reversion + volume spike + choppiness regime filter.
- Primary timeframe: 12h, HTF: 1d for trend and chop regime
- Long: KAMA rising (uptrend) + RSI < 30 (oversold) + volume > 1.8x 20-period avg + CHOP > 61.8 (ranging)
- Short: KAMA falling (downtrend) + RSI > 70 (overbought) + volume > 1.8x 20-period avg + CHOP > 61.8 (ranging)
- Exit: RSI crosses 50 (mean reversion midpoint)
- Uses KAMA for adaptive trend, RSI for mean reversion in ranging markets, volume for confirmation
- Choppiness regime filter ensures we only trade in ranging markets (CHOP > 61.8) where mean reversion works
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to balance return and risk
- BTC/ETH focus: requires HTF trend alignment to avoid SOL-only bias
- Works in ranging markets (2025-2026 bear/range) where mean reversion excels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.8x 20-period average (volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d KAMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # KAMA calculation (adaptive moving average)
    def calculate_kama(price, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=0)
        # Handle first period values
        er = np.zeros_like(price)
        er[period:] = change[period-1:] / np.maximum(volatility[period-1:], 1e-10)
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama = np.zeros_like(price)
        kama[0] = price[0]
        for i in range(1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    # Calculate 1d RSI for mean reversion
    def calculate_rsi(price, period=14):
        delta = np.diff(price)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(price)
        avg_loss = np.zeros_like(price)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(price)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate 1d Choppiness Index for regime filter
    def calculate_choppiness(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # Align with close
        
        # ATR
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            if i < period:
                atr[i] = np.nan
            else:
                atr[i] = np.nanmean(tr[i-period+1:i+1])
        
        # Sum of ATR over period
        atr_sum = np.zeros_like(close)
        for i in range(period-1, len(close)):
            atr_sum[i] = np.nansum(atr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        hh = np.zeros_like(close)
        ll = np.zeros_like(close)
        for i in range(len(close)):
            if i < period-1:
                hh[i] = np.nan
                ll[i] = np.nan
            else:
                hh[i] = np.nanmax(high[i-period+1:i+1])
                ll[i] = np.nanmin(low[i-period+1:i+1])
        
        # Choppiness Index
        chop = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if atr_sum[i] > 0 and hh[i] != ll[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
            else:
                chop[i] = 50  # Neutral when undefined
        return chop
    
    # Calculate indicators on 1d data
    kama_1d = calculate_kama(close_1d)
    rsi_1d = calculate_rsi(close_1d)
    chop_1d = calculate_choppiness(high_1d, low_1d, close_1d)
    
    # Align to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30)  # Need 20 for volume MA, 30 for KAMA/RSI/CHOP
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.8x average)
        volume_spike = volume[i] > 1.8 * vol_ma[i]
        
        # Choppiness regime: > 61.8 = ranging (mean reversion zone)
        ranging_market = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: KAMA rising (uptrend) + RSI < 30 (oversold) + volume spike + ranging market
            if (i > 0 and kama_aligned[i] > kama_aligned[i-1] and  # KAMA rising
                rsi_aligned[i] < 30 and 
                volume_spike and 
                ranging_market):
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling (downtrend) + RSI > 70 (overbought) + volume spike + ranging market
            elif (i > 0 and kama_aligned[i] < kama_aligned[i-1] and  # KAMA falling
                  rsi_aligned[i] > 70 and 
                  volume_spike and 
                  ranging_market):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI crosses above 50 (mean reversion to midpoint)
            if rsi_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI crosses below 50 (mean reversion to midpoint)
            if rsi_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_RSI_VolumeSpike_Choppiness"
timeframe = "12h"
leverage = 1.0