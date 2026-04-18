#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume Confirmation and ADX Trend Filter
Hypothesis: Weekly Donchian channels capture major trend shifts. On daily timeframe,
we break out of the prior week's Donchian bands with volume confirmation and ADX
trend strength filter to avoid false breakouts. This low-frequency approach
minimizes fee drag while capturing sustained moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian(20) - using weekly high/low
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Calculate 20-period rolling max/min on weekly data
    donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (will only update after weekly bar closes)
    donchian_high_daily = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_daily = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # ADX(14) for trend strength on daily data
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume filter: current volume > 1.3x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 25  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_daily[i]) or np.isnan(donchian_low_daily[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high_daily[i]
        lower = donchian_low_daily[i]
        trend_strength = adx[i]
        vol_ok = vol_filter[i]
        
        # Only trade when trend is strong enough (ADX > 25)
        if trend_strength < 25:
            # Weak trend - stay flat or exit
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above weekly Donchian high with volume
            if price > upper and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below weekly Donchian low with volume
            elif price < lower and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long: exit if price returns to weekly Donchian low or trend weakens
            if price < lower or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short: exit if price returns to weekly Donchian high or trend weakens
            if price > upper or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_ADX"
timeframe = "1d"
leverage = 1.0