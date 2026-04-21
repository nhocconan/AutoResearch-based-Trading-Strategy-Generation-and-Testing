#!/usr/bin/env python3
"""
4h_HTFTrend_RegimeFilter_VolumeBreakout
Hypothesis: 4h price breakouts above/below rolling Donchian(20) channels, filtered by 1d EMA50 trend and 1w ADX regime (trending > 25), with volume confirmation (>1.5x 20-bar average). Uses ATR(14) trailing stop (2.0*ATR) for exits. Designed for low trade frequency (target: 20-40/year) to minimize fee drag. Works in bull/bear via HTF trend alignment and volatility regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (daily for EMA trend, weekly for ADX regime)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 30:
        return np.zeros(n)
    
    # === Daily EMA50 for HTF trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Weekly ADX(14) for regime filter (trending > 25) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = pd.Series(high_1w - low_1w)
    tr2 = pd.Series(np.abs(high_1w - np.roll(close_1w, 1)))
    tr3 = pd.Series(np.abs(low_1w - np.roll(close_1w, 1)))
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                                 np.maximum(high_1w - np.roll(high_1w, 1), 0), 0))
    dm_minus = pd.Series(np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                                  np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0))
    
    # Smoothed values
    tr_14 = tr_1w.ewm(alpha=1/14, adjust=False).mean()
    dm_plus_14 = dm_plus.ewm(alpha=1/14, adjust=False).mean()
    dm_minus_14 = dm_minus.ewm(alpha=1/14, adjust=False).mean()
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1w = dx.ewm(alpha=1/14, adjust=False).mean()
    adx_1w_values = adx_1w.values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w_values)
    
    # === 4h Donchian(20) breakout channels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian channels: upper = max(high, lookback=20), lower = min(low, lookback=20)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation (>1.5x 20-bar average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx_1w_aligned[i]) 
            or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = volume[i] > 1.5 * vol_ma[i]
            
            # Regime filter: weekly ADX > 25 (trending market)
            regime_filter = adx_1w_aligned[i] > 25
            
            # Long conditions: price > Donchian upper, price > daily EMA50, volume spike, trending regime
            long_breakout = price > donch_high[i]
            long_trend = price > ema_50_1d_aligned[i]
            
            # Short conditions: price < Donchian lower, price < daily EMA50, volume spike, trending regime
            short_breakout = price < donch_low[i]
            short_trend = price < ema_50_1d_aligned[i]
            
            # Entry logic
            if long_breakout and long_trend and vol_confirm and regime_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_confirm and regime_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below Donchian lower (support broken)
            elif price < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above Donchian upper (resistance broken)
            elif price > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTFTrend_RegimeFilter_VolumeBreakout"
timeframe = "4h"
leverage = 1.0