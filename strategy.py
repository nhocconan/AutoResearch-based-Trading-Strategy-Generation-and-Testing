#!/usr/bin/env python3
"""
1h_4H_1D_DonchianBreakout_VolumeChopFilter_V1
Hypothesis: 1h Donchian channel breakouts with 4h/1d trend alignment and volume/Chop regime filter.
- Uses 4h EMA50 and 1d EMA200 for trend filter (avoid counter-trend trades)
- Uses 4h ATR(14) normalized by price for volatility filter
- Uses 1h Chop index (14) to avoid ranging markets (Chop > 61.8 = range, < 38.2 = trend)
- Requires volume > 1.5x 20-period MA for breakout confirmation
- Position size: 0.20 (discrete to minimize fee churn)
- Target: 15-37 trades/year (60-150 over 4 years) by using tight entry conditions
- Works in bull/bear via trend filter and volatility-adjusted breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === Load HTF data ONCE before loop ===
    # 4h for EMA50 trend and ATR volatility
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 1d for EMA200 long-term trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # === 4h Indicators ===
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 4h EMA50 for medium-term trend
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 4h ATR(14) for volatility normalization
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    
    # 4h Volume MA(20) for volume filter
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # === 1d Indicators ===
    close_1d = df_1d['close'].values
    # 1d EMA200 for long-term trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 1h Indicators (primary timeframe) ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1h Chop index (14) - measures ranging vs trending
    tr_1h = pd.Series(np.maximum(
        high - low,
        np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    ))
    atr_1h_14 = tr_1h.rolling(window=14, min_periods=14).mean().values
    highest_high_1h_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_1h_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_1h_14 - lowest_low_1h_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop = 100 * np.log10(atr_1h_14 * np.sqrt(14) / chop_denom) / np.log10(10)
    # Handle NaN from log calculations
    chop = np.where(np.isnan(chop), 50.0, chop)
    
    # 1h Volume MA(20) for breakout confirmation
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if indicators not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_ma_4h_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(chop[i]) or np.isnan(vol_ma_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma_1h[i]  # volume confirmation for breakout
        
        # Trend filters: 4h EMA50 and 1d EMA200 alignment
        uptrend = price > ema_50_4h_aligned[i] and price > ema_200_1d_aligned[i]
        downtrend = price < ema_50_4h_aligned[i] and price < ema_200_1d_aligned[i]
        
        # Volatility filter: avoid extremely low/high volatility
        vol_norm = atr_14_aligned[i] / price if price > 0 else 0
        vol_filter = 0.005 < vol_norm < 0.05  # reasonable ATR% range
        
        # Chop filter: only trade when trending (Chop < 38.2) or moderately ranging (38.2 <= Chop <= 61.8)
        chop_filter = chop[i] < 61.8  # avoid strong ranging markets
        
        if position == 0:
            # Long: price breaks above Donchian high + volume + trend + vol + chop filters
            if price > donchian_high[i] and vol_ok and uptrend and vol_filter and chop_filter:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Donchian low + volume + trend + vol + chop filters
            elif price < donchian_low[i] and vol_ok and downtrend and vol_filter and chop_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend reverses
            if price < donchian_low[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend reverses
            if price > donchian_high[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4H_1D_DonchianBreakout_VolumeChopFilter_V1"
timeframe = "1h"
leverage = 1.0