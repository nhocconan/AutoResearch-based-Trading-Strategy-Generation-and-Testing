#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining 1-week RSI mean reversion with 1-day ADX trend filter.
# In strong trends (1-day ADX > 25), avoid mean reversion to prevent whipsaws.
# In weak trends or ranging markets (1-day ADX <= 25), use 1-week RSI < 30 for long and > 70 for short.
# Weekly RSI avoids short-term noise and captures longer-term overextensions.
# Volume confirmation: current volume > 1.5x 20-period average to ensure participation.
# Designed to work in both bull and bear markets by adapting to trend strength.
# Target: 20-50 trades/year per symbol (80-200 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    # Load 1w data ONCE for RSI
    df_1w = get_htf_data(prices, '1w')
    
    # 1-day ADX(14) for trend strength
    adx_len = 14
    if len(df_1d) < adx_len:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1-week RSI(14)
    rsi_len = 14
    if len(df_1w) < rsi_len:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w)
    delta = np.concatenate([[np.nan], delta])
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_len, adjust=False, min_periods=rsi_len).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_len, adjust=False, min_periods=rsi_len).mean().values
    
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, np.inf, rs)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(adx_len*2, rsi_len*2, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend regime: ADX > 25 = trending, ADX <= 25 = ranging/weak trend
        strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            if not strong_trend and volume_confirmed:
                # In weak trend or ranging: use weekly RSI for mean reversion
                if rsi_aligned[i] < 30:
                    position = 1
                    signals[i] = position_size
                elif rsi_aligned[i] > 70:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                # In strong trend: avoid mean reversion trades
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or becomes overbought
            if rsi_aligned[i] >= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or becomes oversold
            if rsi_aligned[i] <= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1wRSI_1dADX_MeanRev_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0