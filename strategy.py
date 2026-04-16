#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-day RSI mean-reversion + 1-day volume surge + 1-week ADX trend filter.
# RSI identifies overbought/oversold conditions, volume surge confirms institutional participation,
# ADX ensures trades occur in trending markets. Works in bull/bear by fading extremes in ranges
# and catching momentum in trends. Target: 50-150 total trades over 4 years (12-37/year). Size: 0.25.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1-day RSI (14) for mean-reversion ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # RSI: 100 - (100 / (1 + RS))
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    avg_gain = wilders_smoothing(gain, 14)
    avg_loss = wilders_smoothing(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where((avg_gain == 0) & (avg_loss == 0), 50, rsi)
    
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === 1-day volume for surge confirmation ===
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # === 1-week ADX (14) for trend strength ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Wilder's smoothing
    atr_14 = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_14
    di_minus = 100 * dm_minus_smooth / atr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx_14 = wilders_smoothing(dx, 14)
    
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_14)
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for weekly indicators
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        
        # Volume surge: current 1d volume > 2.0x 20-period average
        df_1d_current = get_htf_data(prices, '1d')
        vol_1d_current = df_1d_current['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d_current, vol_1d_current)
        vol_surge = vol_1d_aligned[i] > vol_ma_20_aligned[i] * 2.0
        
        # Trend filter: ADX > 20 indicates trending market
        trending = adx_1w_aligned[i] > 20.0
        
        # RSI levels: oversold < 30, overbought > 70
        oversold = rsi_1d_aligned[i] < 30
        overbought = rsi_1d_aligned[i] > 70
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: RSI oversold + volume surge + trending
            if oversold and vol_surge and trending:
                signals[i] = 0.25
                position = 1
                continue
            # Short: RSI overbought + volume surge + trending
            elif overbought and vol_surge and trending:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal on opposite RSI extreme
        elif position == 1:
            # Exit long if RSI becomes overbought
            if overbought:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if RSI becomes oversold
            if oversold:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_RSI14_1dVolume2.0x_1wADX20_TrendFilter"
timeframe = "12h"
leverage = 1.0