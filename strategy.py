#!/usr/bin/env python3
"""
6h_1d_Adaptive_RSI_Range_Breakout
Hypothesis: Combine RSI mean reversion in ranging markets with breakout momentum in trending markets.
Uses daily ADX to detect regime: ADX<20 = range (fade RSI extremes), ADX>25 = trend (breakout RSI pullbacks).
In range: long RSI<30, short RSI>70. In trend: long RSI>50 & rising, short RSI<50 & falling.
Adds 6h volume confirmation to avoid false signals. Designed for low trade frequency (~15-30/year) with adaptive logic.
Works in bull markets (trend breakouts) and bear markets (range mean reversion).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Adaptive_RSI_Range_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR REGIME DETECTION ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        atr[period-1] = np.nanmean(tr[:period])
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
            
            if atr[i] != 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
            else:
                plus_di[i] = 0
                minus_di[i] = 0
        
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        for i in range(2*period-1, len(high)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
            else:
                dx[i] = 0
        
        adx[2*period-1] = np.nanmean(dx[period-1:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6H INDICATORS ===
    # RSI(14)
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        avg_gain[period-1] = np.nanmean(gain[:period])
        avg_loss[period-1] = np.nanmean(loss[:period])
        
        for i in range(period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.zeros_like(close)
        rsi = np.zeros_like(close)
        for i in range(period, len(close)):
            if avg_loss[i] != 0:
                rs[i] = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs[i]))
            else:
                rsi[i] = 100
        
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        adx = adx_aligned[i]
        rsi_val = rsi[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Regime detection
        is_ranging = adx < 20
        is_trending = adx > 25
        
        # Volume filter: require at least 1.5x average volume
        volume_ok = vol_ratio >= 1.5
        
        # Initialize signal
        long_signal = False
        short_signal = False
        
        if is_ranging and volume_ok:
            # Mean reversion in ranging market
            long_signal = rsi_val < 30
            short_signal = rsi_val > 70
        elif is_trending and volume_ok:
            # Momentum in trending market
            # Long: RSI > 50 and rising (above previous)
            # Short: RSI < 50 and falling (below previous)
            if i > 0:
                rsi_prev = rsi[i-1]
                long_signal = rsi_val > 50 and rsi_val > rsi_prev
                short_signal = rsi_val < 50 and rsi_val < rsi_prev
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long: RSI > 70 (overbought) or regime shift to ranging with RSI > 50
            exit_long = rsi_val > 70 or (is_ranging and rsi_val > 50)
        elif position == -1:
            # Exit short: RSI < 30 (oversold) or regime shift to ranging with RSI < 50
            exit_short = rsi_val < 30 or (is_ranging and rsi_val < 50)
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals