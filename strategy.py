#!/usr/bin/env python3
"""
Hypothesis: 12h ADX trend filter + 1d RSI mean reversion + volume spike.
In trending markets (ADX > 25): trade RSI pullbacks in direction of trend.
In ranging markets (ADX <= 25): trade RSI extremes for mean reversion.
Volume must be above 30-period average to confirm signals.
Uses weekly trend filter only for regime confirmation, not direct trading.
Target: 50-150 total trades over 4 years with controlled risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_adx_rsi_volume_spike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY TREND FILTER (REGIME) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # === DAILY ADX FOR TREND STRENGTH ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = d_high[1:] - d_low[1:]
    tr2 = np.abs(d_high[1:] - d_close[:-1])
    tr3 = np.abs(d_low[1:] - d_close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # Directional Movement
    dm_plus = np.where((d_high[1:] - d_high[:-1]) > (d_low[:-1] - d_low[1:]), 
                       np.maximum(d_high[1:] - d_high[:-1], 0), 0)
    dm_minus = np.where((d_low[:-1] - d_low[1:]) > (d_high[1:] - d_high[:-1]), 
                        np.maximum(d_low[:-1] - d_low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def smoothed_avg(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(x[1:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(x)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + x[i]) / period
        return result
    
    atr = smoothed_avg(tr, 14)
    dm_plus_smooth = smoothed_avg(dm_plus, 14)
    dm_minus_smooth = smoothed_avg(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, dm_plus_smooth / atr * 100, 0)
    di_minus = np.where(atr != 0, dm_minus_smooth / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = smoothed_avg(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === DAILY RSI (14) ===
    delta = np.diff(d_close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(d_close, np.nan)
    avg_loss = np.full_like(d_close, np.nan)
    
    # Wilder smoothing for RSI
    period = 14
    if len(gain) >= period:
        avg_gain[period] = np.nanmean(gain[1:period+1])
        avg_loss[period] = np.nanmean(loss[1:period+1])
        for i in range(period+1, len(d_close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === VOLUME SPIKE (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        if (np.isnan(weekly_ema_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime
        trending = adx_aligned[i] > 25
        bull_trend = close[i] > weekly_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI overbought OR trend reversal
            if rsi_aligned[i] > 70 or (trending and not bull_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI oversold OR trend reversal
            if rsi_aligned[i] < 30 or (trending and bull_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            if trending:
                # Trending market: trade pullbacks in trend direction
                if bull_trend and rsi_aligned[i] < 40:  # Pullback in uptrend
                    position = 1
                    signals[i] = 0.25
                elif not bull_trend and rsi_aligned[i] > 60:  # Pullback in downtrend
                    position = -1
                    signals[i] = -0.25
            else:
                # Ranging market: mean reversion at extremes
                if rsi_aligned[i] < 30:  # Oversold -> long
                    position = 1
                    signals[i] = 0.25
                elif rsi_aligned[i] > 70:  # Overbought -> short
                    position = -1
                    signals[i] = -0.25
    
    return signals