#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using daily Donchian channel breakout with 1d EMA34 trend filter and volume confirmation.
- Long when price closes above daily Donchian upper (20) + volume > 1.8x 20-period 12h volume MA + price above 1d EMA34
- Short when price closes below daily Donchian lower (20) + volume > 1.8x 20-period 12h volume MA + price below 1d EMA34
- Fixed position size 0.25 to limit fee churn and manage drawdown
- ATR-based trailing stop (2.0x ATR) to lock in profits
- Daily Donchian and EMA34 derived from prior day's data (no look-ahead)
- Designed for very low trade frequency (target: 12-37 trades/year) to avoid fee drag
- Works in bull markets (buying above daily Donchian upper with 1d EMA34 uptrend) and bear markets (selling below daily Donchian lower with 1d EMA34 downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channel and EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Donchian channel (20-period)
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all HTF indicators to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 12h data for volume confirmation and ATR (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Volume average (20-period) on 12h for confirmation
    volume_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (10-period) on 12h for stoploss
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Align all indicators to 12h timeframe (primary)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr_10)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and 1d EMA34 trend filter
            # Long: price closes above daily Donchian upper + volume spike + price above 1d EMA34
            if price > donch_high_val and vol > 1.8 * vol_ma and price > ema_34_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.0 * atr_val
            # Short: price closes below daily Donchian lower + volume spike + price below 1d EMA34
            elif price < donch_low_val and vol > 1.8 * vol_ma and price < ema_34_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_stop = entry_price + 2.0 * atr_val
        
        elif position == 1:
            # Check stoploss
            if price <= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail stop: raise stop if price moves favorably
                atr_stop = max(atr_stop, price - 1.5 * atr_val)
        
        elif position == -1:
            # Check stoploss
            if price >= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stop: lower stop if price moves favorably
                atr_stop = min(atr_stop, price + 1.5 * atr_val)
    
    return signals

name = "12h_Donchian20_1dEMA34_VolumeSpike_ATRTrail"
timeframe = "12h"
leverage = 1.0