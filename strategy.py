#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Uses 4h for signal direction (EMA50) and 1d for regime filter (ADX > 25)
# 1h only for entry timing precision: long when price breaks above H3 with volume spike in uptrend,
# short when price breaks below L3 with volume spike in downtrend
# Designed for low trade frequency: ~20-40 trades/year per symbol with 0.20 sizing
# Camarilla pivot levels provide institutional support/resistance; volume spike confirms participation
# Works in bull/bear markets by following 4h trend direction and avoiding low-volatility regimes

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_Trend_1dADX_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h HTF data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d HTF data for regime filter (ADX > 25)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
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
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1h data for Camarilla pivot calculation (using previous day's OHLC)
    # We'll use rolling window to get previous day's high, low, close
    # Since we don't have daily data aligned perfectly, we approximate with 24-period lookback
    # But better approach: use the actual 1d data we already loaded
    # Camarilla levels: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    high_1d_val = df_1d['high'].values
    low_1d_val = df_1d['low'].values
    close_1d_val = df_1d['close'].values
    
    camarilla_range = (high_1d_val - low_1d_val) * 1.1 / 2
    h3_level = close_1d_val + camarilla_range
    l3_level = close_1d_val - camarilla_range
    
    # Align Camarilla levels to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_level)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_level)
    
    # Volume confirmation: volume > 1.3 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ema_20)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(60, 30, 20)  # Need 4h EMA50, 1d ADX, volume EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ema_20[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when ADX > 25 (trending market)
        if adx_aligned[i] <= 25:
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: direction of 4h EMA50
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if uptrend:
                # Long: price breaks above H3 with volume spike
                if close[i] > h3_aligned[i] and volume_spike[i]:
                    signals[i] = 0.20
                    position = 1
                else:
                    signals[i] = 0.0
            elif downtrend:
                # Short: price breaks below L3 with volume spike
                if close[i] < l3_aligned[i] and volume_spike[i]:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid sideways markets
        
        elif position == 1:  # Long position
            # Exit: price breaks below L3 (mean reversion) or loss of momentum
            if close[i] < l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price breaks above H3 (mean reversion) or loss of momentum
            if close[i] > h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals