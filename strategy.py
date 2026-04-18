#!/usr/bin/env python3
"""
4h_RSI_Extreme_Trend_Filter_v1
Hypothesis: RSI extremes (overbought/oversold) combined with 4h trend filter (EMA50) provide high-probability mean reversion entries in both bull and bear markets.
Long when RSI < 25 and price > EMA50 (bullish bias during pullback).
Short when RSI > 75 and price < EMA50 (bearish bias during bounce).
Only trade when 12h ADX < 30 to avoid strong trends where mean reversion fails.
Target: 20-40 trades/year by using strict RSI thresholds and trend filter.
Works in ranging markets via mean reversion and avoids losses in strong trends via ADX filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate RSI(14) on close
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Use Wilder's smoothing (alpha = 1/period)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        if len(close) > period:
            # Initial average
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            # Wilder smoothing
            for i in range(period + 1, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # EMA50 for trend filter
    if len(close) >= 50:
        ema_50 = pd.Series(close).ewm(span=50, adjust=False).mean().values
    else:
        ema_50 = np.full_like(close, np.nan)
    
    # Get 12h data for ADX filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 14-period ADX for regime filtering
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smooth TR, DM+
        atr = np.full_like(tr, np.nan)
        dm_plus_smooth = np.full_like(dm_plus, np.nan)
        dm_minus_smooth = np.full_like(dm_minus, np.nan)
        
        if len(tr) >= period:
            # Initial values
            atr[period] = np.nanmean(tr[1:period+1])
            dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
            dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
            
            # Wilder smoothing
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # DI+ and DI-
        di_plus = np.full_like(dm_plus_smooth, np.nan)
        di_minus = np.full_like(dm_minus_smooth, np.nan)
        valid = ~np.isnan(atr) & (atr != 0)
        di_plus[valid] = 100 * dm_plus_smooth[valid] / atr[valid]
        di_minus[valid] = 100 * dm_minus_smooth[valid] / atr[valid]
        
        # DX and ADX
        dx = np.full_like(di_plus, np.nan)
        dx_valid = ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0)
        dx[dx_valid] = 100 * np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / (di_plus[dx_valid] + di_minus[dx_valid])
        
        adx = np.full_like(dx, np.nan)
        if len(dx) >= period:
            # Initial ADX
            adx[2*period-1] = np.nanmean(dx[period:2*period])
            # Wilder smoothing for ADX
            for i in range(2*period, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # Align 12h ADX to 4h timeframe
    adx_4h = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14) + 1  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(ema_50[i]) or 
            np.isnan(adx_4h[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to EMA50
        price_above_ema = close[i] > ema_50[i]
        price_below_ema = close[i] < ema_50[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 25
        rsi_overbought = rsi[i] > 75
        
        # Regime filter: only trade in low volatility (ADX < 30)
        low_volatility = adx_4h[i] < 30
        
        if position == 0:
            # Long: RSI oversold + price above EMA50 + low volatility
            if rsi_oversold and price_above_ema and low_volatility:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought + price below EMA50 + low volatility
            elif rsi_overbought and price_below_ema and low_volatility:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (40-60) OR ADX rises above 40 (strong trend)
            if (40 <= rsi[i] <= 60) or adx_4h[i] > 40:
                signals[i] = 0.0  # exit to flat
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (40-60) OR ADX rises above 40 (strong trend)
            if (40 <= rsi[i] <= 60) or adx_4h[i] > 40:
                signals[i] = 0.0  # exit to flat
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Extreme_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0