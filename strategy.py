#!/usr/bin/env python3
# 4h_VWAP_KAMA_BREAKOUT_12H_TREND_V1
# Hypothesis: Use 12h KAMA trend for direction, enter on 4h VWAP breakouts with volume confirmation
# and 12h ADX > 25 for trend strength. KAMA adapts to volatility, reducing whipsaw in chop.
# VWAP breakouts capture institutional flow. Works in bull/bear by following higher timeframe trend.
# Target: 20-50 trades per year (~80-200 total over 4 years).

name = "4h_VWAP_KAMA_BREAKOUT_12H_TREND_V1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h KAMA for trend direction (adaptive)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate Efficiency Ratio and Smoothing Constants for KAMA
    def kama(close, length=10, fast=2, slow=30):
        # Change
        change = np.abs(np.diff(close, prepend=close[0]))
        # Volatility (sum of absolute changes)
        volatility = np.zeros_like(close)
        for i in range(1, len(close)):
            volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
        
        # Avoid division by zero
        er = np.zeros_like(close)
        for i in range(length, len(close)):
            if volatility[i] != 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_12h = kama(close_12h, length=10, fast=2, slow=30)
    # Trend: 1 if close > KAMA, -1 if close < KAMA
    trend_12h = np.where(close_12h > kama_12h, 1, -1)
    
    # 12h ADX for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        if len(tr) > period:
            atr[period-1] = np.mean(tr[1:period])
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Avoid division by zero
        dx = np.zeros_like(close)
        for i in range(len(close)):
            if atr[i] != 0:
                plus_di = 100 * plus_dm[i] / atr[i]
                minus_di = 100 * minus_dm[i] / atr[i]
                if plus_di + minus_di != 0:
                    dx[i] = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Smooth DX
        adx = np.zeros_like(close)
        if len(dx) > period:
            adx[period-1] = np.mean(dx[1:period]) if period > 1 else 0
            for i in range(period, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_12h = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values)
    
    # Align 12h indicators to 4h
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 4h VWAP calculation (session-based: reset daily)
    # Typical price
    typical_price = (high + low + close) / 3.0
    # VWAP = cumulative(typical_price * volume) / cumulative(volume)
    # Reset at midnight UTC (00:00) each day
    vwap = np.full(n, np.nan)
    cum_tp_vol = 0.0
    cum_vol = 0.0
    
    for i in range(n):
        # Reset at 00:00 UTC each day
        if i > 0:
            prev_time = pd.Timestamp(prices['open_time'].iloc[i-1])
            curr_time = pd.Timestamp(prices['open_time'].iloc[i])
            if prev_time.date() != curr_time.date():
                cum_tp_vol = 0.0
                cum_vol = 0.0
        
        cum_tp_vol += typical_price[i] * volume[i]
        cum_vol += volume[i]
        if cum_vol != 0:
            vwap[i] = cum_tp_vol / cum_vol
    
    # Volume spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma != 0, volume / vol_ma, 0)
    vol_spike = vol_ratio > 1.8  # 80% above average volume
    
    # Session filter: 08-20 UTC (avoid low liquidity periods)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(30, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trend_12h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(vwap[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in session and with volume spike
        if not (in_session[i] and vol_spike[i]):
            if position != 0:
                # Hold position until exit signal
                pass
            else:
                signals[i] = 0.0
                continue
        
        if position == 1:  # Long position
            # Exit: 12h trend turns bearish OR ADX weakens OR price closes below VWAP
            if trend_12h_aligned[i] == -1 or adx_12h_aligned[i] < 20 or close[i] < vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: 12h trend turns bullish OR ADX weakens OR price closes above VWAP
            if trend_12h_aligned[i] == 1 or adx_12h_aligned[i] < 20 or close[i] > vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need strong trend (ADX > 25) and price breaks VWAP with volume
            if adx_12h_aligned[i] > 25:
                # Long: price breaks above VWAP on volume
                if close[i] > vwap[i] and close[i-1] <= vwap[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below VWAP on volume
                elif close[i] < vwap[i] and close[i-1] >= vwap[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals