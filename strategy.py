#!/usr/bin/env python3
"""
1d_Donchian20_WeeklyTrend_VolumeSpike_ATRStop
Hypothesis: Daily Donchian(20) breakout with weekly EMA34 trend filter and volume confirmation (>1.5x 20-day volume MA). 
In weekly uptrend (price > weekly EMA34): long Donchian(20) breakout. 
In weekly downtrend (price < weekly EMA34): short Donchian(20) breakdown. 
Volume confirmation reduces false breakouts. ATR(14) trailing stop (2.5x ATR) manages risk. 
Position size 0.25 balances risk/return. Target ~15-30 trades/year per symbol (60-120 total over 4 years).
Uses 1d primary timeframe with 1w HTF for trend filter. Designed to work in both bull (trend following) and bear (short breakdowns) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === 1w Indicators (HTF for trend filter) ===
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1d Indicators (primary timeframe) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        weekly_trend_up = price > ema_34_1w_aligned[i]  # weekly uptrend
        weekly_trend_down = price < ema_34_1w_aligned[i]  # weekly downtrend
        
        if position == 0:
            if weekly_trend_up and vol_ok:
                # Weekly uptrend: long Donchian breakout
                if price > donchian_high[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    highest_since_entry = price
            elif weekly_trend_down and vol_ok:
                # Weekly downtrend: short Donchian breakdown
                if price < donchian_low[i]:
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

name = "1d_Donchian20_WeeklyTrend_VolumeSpike_ATRStop"
timeframe = "1d"
leverage = 1.0