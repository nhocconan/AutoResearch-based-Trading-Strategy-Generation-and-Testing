#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume Spike + 1d EMA Trend + ATR Stop
Hypothesis: Donchian(20) breakouts capture strong momentum, filtered by volume spike and 1d EMA50 trend. 
ATR-based stop loss limits downside. Designed for low frequency (20-50 trades/year) with edge in both bull and bear markets.
"""

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
    
    # Donchian channel (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # ATR for stop loss and volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d EMA50 for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = 50  # need enough history for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema_trend = ema_50_1d_aligned[i]
        volatility = atr[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above Donchian high with volume spike and above daily EMA50
            if price > upper and vol_spike and price > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_at_entry = volatility
            # Short: break below Donchian low with volume spike and below daily EMA50
            elif price < lower and vol_spike and price < ema_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_at_entry = volatility
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: ATR-based stop loss or price retrace to midpoint
            if price <= entry_price - 2.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            elif price <= (upper + lower) / 2.0:  # retrace to channel midpoint
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: ATR-based stop loss or price retrace to midpoint
            if price >= entry_price + 2.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            elif price >= (upper + lower) / 2.0:  # retrace to channel midpoint
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_VolumeSpike_1dEMA50_ATRStop"
timeframe = "4h"
leverage = 1.0