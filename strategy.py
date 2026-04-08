#!/usr/bin/env python3
"""
1h Volume-Weighted RSI Pullback with 4h Trend and 1d Regime Filter
Hypothesis: In trending markets (4h), pullbacks to the 21-period EMA on 1h with 
RSI(14) < 30 (oversold) or > 70 (overbought) provide high-probability mean-reversion 
entries when confirmed by volume spikes. The 1d ADX filter ensures we only trade 
in trending regimes (ADX > 25) to avoid whipsaws in ranging markets. 
Designed for 15-35 trades/year by requiring multiple confluences.
Works in bull markets via pullbacks in uptrends and in bear markets via bounces in downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_volume_weighted_rsi_pullback_4h_trend_1d_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h indicators
    # EMA(21) for pullback target
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # 4h trend filter: EMA(50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d regime filter: ADX(14) > 25 for trending market
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    # Handle division by zero
    dx = np.where((di_plus + di_minus) != 0, dx, 0)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_spike[i]) or
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50 (momentum fading) or price closes below EMA21
            if (rsi[i] > 50 or 
                close[i] < ema_21[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50 or price closes above EMA21
            if (rsi[i] < 50 or 
                close[i] > ema_21[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Trend filter from 4h
            uptrend_4h = close[i] > ema_50_4h_aligned[i]
            downtrend_4h = close[i] < ema_50_4h_aligned[i]
            
            # Regime filter: only trade in trending markets (ADX > 25)
            trending = adx_aligned[i] > 25
            
            # Long: pullback to EMA21 with RSI oversold and volume spike in uptrend
            if (close[i] <= ema_21[i] * 1.005 and  # Allow small buffer above EMA
                rsi[i] < 30 and 
                vol_spike[i] and 
                uptrend_4h and 
                trending):
                position = 1
                signals[i] = 0.20
            # Short: bounce off EMA21 with RSI overbought and volume spike in downtrend
            elif (close[i] >= ema_21[i] * 0.995 and  # Allow small buffer below EMA
                  rsi[i] > 70 and 
                  vol_spike[i] and 
                  downtrend_4h and 
                  trending):
                position = -1
                signals[i] = -0.20
    
    return signals