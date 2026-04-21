#!/usr/bin/env python3
"""
6h_ADX_Regime_Donchian20_Breakout
Hypothesis: On 6h timeframe, use ADX(14) to filter regimes (ADX>25 = trending, ADX<20 = ranging).
In trending regimes, trade Donchian(20) breakouts with volume confirmation (>1.5x 20-period MA).
In ranging regimes, fade at Donchian channels with volume confirmation.
HTF: 12h EMA50 for trend bias (long only above EMA50, short only below).
Discrete sizing (0.25) with ATR(14) stoploss (2.0x). Designed for low trade frequency (~15-25/year).
Works in both bull/bear via ADX regime filter and 12h EMA50 bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for EMA50 trend bias)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # === 12h EMA50 for trend bias ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 6h ADX(14) for regime filter ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high - np.roll(high, 1))
    down_move = pd.Series(np.roll(low, 1) - low)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # === 6h Donchian(20) channels ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Volume confirmation (>1.5x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(adx[i]) or np.isnan(atr[i]) 
            or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        ema_50 = ema_50_12h_aligned[i]
        adx_val = adx[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_conf = volume_now > 1.5 * vol_avg
        
        if position == 0:
            # Regime-based entry logic
            if adx_val > 25:  # Trending regime
                # Breakout continuation with volume and HTF bias
                long_condition = (price > upper_channel) and (price > ema_50) and volume_conf
                short_condition = (price < lower_channel) and (price < ema_50) and volume_conf
                
                if long_condition:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif short_condition:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    
            elif adx_val < 20:  # Ranging regime
                # Fade at channels with volume and HTF bias (counter-trend)
                long_condition = (price < lower_channel) and (price > ema_50) and volume_conf
                short_condition = (price > upper_channel) and (price < ema_50) and volume_conf
                
                if long_condition:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif short_condition:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (price below EMA50)
            elif price < ema_50:
                signals[i] = 0.0
                position = 0
            # Channel reversion exit (price back to mid-channel)
            elif price > (upper_channel + lower_channel) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (price above EMA50)
            elif price > ema_50:
                signals[i] = 0.0
                position = 0
            # Channel reversion exit (price back to mid-channel)
            elif price < (upper_channel + lower_channel) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_Regime_Donchian20_Breakout"
timeframe = "6h"
leverage = 1.0