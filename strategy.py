#!/usr/bin/env python3
"""
12h_1w_1d_PriceAction_Confluence
Hypothesis: Price action confluence between 1w trend (EMA34), 1d support/resistance (ATR-based), and 12h momentum (RSI) with volume confirmation. Works in bull/bear by only taking trades aligned with higher timeframe trend. Target: 15-30 trades/year with position size 0.25.
"""

name = "12h_1w_1d_PriceAction_Confluence"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly EMA34 for trend
    def calculate_ema(data, period):
        ema = np.full_like(data, np.nan)
        if len(data) < period:
            return ema
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        return ema
    
    ema34_1w = calculate_ema(close_1w, 34)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily ATR for support/resistance levels
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        atr = np.full_like(close, np.nan)
        for i in range(len(tr)):
            if i < period:
                if i == 0:
                    atr[i] = tr[i]
                else:
                    atr[i] = np.mean(tr[:i+1])
            else:
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily pivot points (simple: previous day high/low/close)
    def calculate_pivot_points(high, low, close):
        # Previous day's values
        prev_high = np.roll(high, 1)
        prev_low = np.roll(low, 1)
        prev_close = np.roll(close, 1)
        
        # Pivot point
        pivot = (prev_high + prev_low + prev_close) / 3
        
        # Support and resistance levels
        r1 = 2 * pivot - prev_low
        s1 = 2 * pivot - prev_high
        r2 = pivot + (prev_high - prev_low)
        s2 = pivot - (prev_high - prev_low)
        
        return pivot, r1, r2, s1, s2
    
    pivot_1d, r1_1d, r2_1d, s1_1d, s2_1d = calculate_pivot_points(high_1d, low_1d, close_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 12h RSI for momentum
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        
        for i in range(len(close)):
            if i < period:
                if i == 0:
                    avg_gain[i] = gain[i]
                    avg_loss[i] = loss[i]
                else:
                    avg_gain[i] = np.mean(gain[:i+1])
                    avg_loss[i] = np.mean(loss[:i+1])
            else:
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_12h = calculate_rsi(close, 14)
    
    # Volume ratio (current vs 20-period average)
    def calculate_volume_ratio(volume, period=20):
        vol_ma = np.full_like(volume, np.nan)
        for i in range(len(volume)):
            if i < period:
                if i == 0:
                    vol_ma[i] = volume[i]
                else:
                    vol_ma[i] = np.mean(volume[:i+1])
            else:
                vol_ma[i] = np.mean(volume[i-period+1:i+1])
        return np.divide(volume, vol_ma, out=np.ones_like(volume), where=vol_ma!=0)
    
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(rsi_12h[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long conditions: price above weekly EMA, near support, bullish momentum, volume
            price_above_weekly_ema = close[i] > ema34_1w_aligned[i]
            near_support = (close[i] <= s1_1d_aligned[i] * 1.02) or (close[i] <= pivot_1d_aligned[i] * 1.01)
            bullish_momentum = rsi_12h[i] > 50 and rsi_12h[i] < 70
            volume_confirm = vol_ratio[i] > 1.2
            
            if price_above_weekly_ema and near_support and bullish_momentum and volume_confirm:
                signals[i] = 0.25
                position = 1
            
            # Short conditions: price below weekly EMA, near resistance, bearish momentum, volume
            elif not price_above_weekly_ema and (close[i] >= r1_1d_aligned[i] * 0.98 or close[i] >= pivot_1d_aligned[i] * 0.99) and \
                 rsi_12h[i] < 50 and rsi_12h[i] > 30 and vol_ratio[i] > 1.2:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly EMA OR RSI overbought OR volume dries up
            if close[i] <= ema34_1w_aligned[i] * 0.995 or rsi_12h[i] >= 75 or vol_ratio[i] < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly EMA OR RSI oversold OR volume dries up
            if close[i] >= ema34_1w_aligned[i] * 1.005 or rsi_12h[i] <= 25 or vol_ratio[i] < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals