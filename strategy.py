#!/usr/bin/env python3
"""
12h_Donchian_Breakout_1dEMA50_VolumeSpike_ATRStop
Hypothesis: Donchian(20) breakouts on 12h timeframe with 1d EMA50 trend filter and volume spike capture strong momentum moves. 
ATR-based stop loss limits downside. Designed for low trade frequency (15-25/year) to minimize fee drag while capturing major trends in both bull and bear markets.
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
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels (once before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high']
    low_12h = df_12h['low']
    close_12h = df_12h['close']
    
    # Calculate Donchian(20) on 12h
    high_max_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high = align_htf_to_ltf(prices, df_12h, high_max_20)
    donchian_low = align_htf_to_ltf(prices, df_12h, low_min_20)
    
    # Get 1d data for EMA50 trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close']
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR for volatility and stop calculation (14-period on 12h)
    high_low = high_12h - low_12h
    high_close = np.abs(high_12h - np.roll(close_12h, 1))
    low_close = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_12h, atr_14)
    
    # Volume spike: volume > 2.5 * 30-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema_trend = ema_50_1d_aligned[i]
        atr_now = atr_14_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above upper Donchian with 1d uptrend and volume spike
            if price > upper and price > ema_trend and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_at_entry = atr_now
            # Short: break below lower Donchian with 1d downtrend and volume spike
            elif price < lower and price < ema_trend and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_at_entry = atr_now
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: ATR-based stop or trend reversal
            if price <= entry_price - 2.0 * atr_at_entry or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: ATR-based stop or trend reversal
            if price >= entry_price + 2.0 * atr_at_entry or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian_Breakout_1dEMA50_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0