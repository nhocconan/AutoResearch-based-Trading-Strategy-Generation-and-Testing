#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_PriceChannel_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and price channel (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter (slow EMA for trend direction)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ATR(14) for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 4h Donchian channel (20-period) for breakout signals
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure all indicators are ready
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_aligned[i]) or np.isnan(atr_14_aligned[i]) or \
           np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = vol > 1.8 * vol_ma
        
        # Trend filter: price above/below 1d EMA34
        uptrend = price > ema_34_aligned[i]
        downtrend = price < ema_34_aligned[i]
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_14_aligned[i] > 0.01 * price  # ATR > 1% of price
        
        if position == 0:
            # Long: Price breaks above 4h Donchian high + uptrend + volume
            if price > high_20[i] and uptrend and volume_confirmed and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 4h Donchian low + downtrend + volume
            elif price < low_20[i] and downtrend and volume_confirmed and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below 4h Donchian middle or trend reverses
            mid_20 = (high_20[i] + low_20[i]) / 2.0
            if price < mid_20[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above 4h Donchian middle or trend reverses
            mid_20 = (high_20[i] + low_20[i]) / 2.0
            if price > mid_20[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals