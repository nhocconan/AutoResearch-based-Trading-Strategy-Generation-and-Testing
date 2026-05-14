#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSpike_HTFTrend_ATRStop
Hypothesis: 4h Donchian channel breakout (20-bar) with volume confirmation (>2.0x 20-bar volume MA) and 12h EMA34 trend filter. 
ATR trailing stop (2.5x ATR) manages risk. Works in bull via upper band breakouts, in bear via lower band breakdowns.
Position size 0.25 balances risk/return. Target ~25-60 trades/year per symbol (100-240 total over 4 years).
Uses 4h primary timeframe with 12h HTF for trend alignment, avoiding overtrading while capturing multi-day moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for trend filter)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # === 12h EMA34 for trend filter ===
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 4h Indicators (primary timeframe) ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) from completed previous bar
    # Upper = max(high of last 20 completed bars), Lower = min(low of last 20 completed bars)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    # Shift by 1 to use only completed bars (don't include current bar)
    upper_channel = np.roll(highest_high, 1)
    lower_channel = np.roll(lowest_low, 1)
    upper_channel[0] = np.nan
    lower_channel[0] = np.nan
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # volume confirmation (strict to reduce trades)
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume confirmation + price > 12h EMA34
            if price > upper_channel[i-1] and vol_ok and price > ema_34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: price breaks below lower Donchian + volume confirmation + price < 12h EMA34
            elif price < lower_channel[i-1] and vol_ok and price < ema_34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Update highest since entry
            highest_since_entry = max(highest_since_entry, price)
            # ATR trailing stop: exit if price drops 2.5*ATR from highest since entry
            if price < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest since entry
            lowest_since_entry = min(lowest_since_entry, price)
            # ATR trailing stop: exit if price rises 2.5*ATR from lowest since entry
            if price > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_HTFTrend_ATRStop"
timeframe = "4h"
leverage = 1.0