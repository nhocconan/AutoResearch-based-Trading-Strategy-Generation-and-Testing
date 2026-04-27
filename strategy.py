# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA(34) for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily ATR(14) for volatility filter
    tr1 = pd.Series(df_1d['high'] - df_1d['low'])
    tr2 = pd.Series(np.abs(df_1d['high'] - df_1d['close'].shift(1)))
    tr3 = pd.Series(np.abs(df_1d['low'] - df_1d['close'].shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = 0
    atr14_1d = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Weekly ATR(14) for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    tr1_w = pd.Series(df_1w['high'] - df_1w['low'])
    tr2_w = pd.Series(np.abs(df_1w['high'] - df_1w['close'].shift(1)))
    tr3_w = pd.Series(np.abs(df_1w['low'] - df_1w['close'].shift(1)))
    tr_w = pd.concat([tr1_w, tr2_w, tr3_w], axis=1).max(axis=1)
    tr_w.iloc[0] = 0
    atr14_1w = tr_w.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr14_1w)
    
    # 12h Donchian(20) breakout levels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.30   # Position size: 30% of capital
    
    # Warmup: need enough data for Donchian, volume MA, and HTF indicators
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or np.isnan(atr14_1w_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        
        ema34_val = ema34_1d_aligned[i]
        atr14_1d_val = atr14_1d_aligned[i]
        atr14_1w_val = atr14_1w_aligned[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        vol_spike_val = vol_spike[i]
        
        # Volatility regime filter: use weekly ATR to detect high/low volatility
        # High volatility: weekly ATR > 1.5 * daily ATR (trending market)
        # Low volatility: weekly ATR <= 1.5 * daily ATR (ranging market)
        vol_ratio = atr14_1w_val / atr14_1d_val if atr14_1d_val > 0 else 1.0
        high_vol_regime = vol_ratio > 1.5
        
        if position == 0:
            # Long: price breaks above Donchian high + volume spike + bullish trend (price > daily EMA34) + high volatility regime
            if close[i] > donchian_high_val and close[i-1] <= donchian_high_val and vol_spike_val and close[i] > ema34_val and high_vol_regime:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low + volume spike + bearish trend (price < daily EMA34) + high volatility regime
            elif close[i] < donchian_low_val and close[i-1] >= donchian_low_val and vol_spike_val and close[i] < ema34_val and high_vol_regime:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or volatility drops (low volatility regime)
            if close[i] < donchian_low_val or not high_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian high or volatility drops (low volatility regime)
            if close[i] > donchian_high_val or not high_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_VolumeTrend_Regime_v1"
timeframe = "12h"
leverage = 1.0