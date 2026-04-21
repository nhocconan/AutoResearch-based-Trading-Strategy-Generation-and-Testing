#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_ATRStop_v1
Hypothesis: Daily Camarilla pivot (R1/S1) breakout filtered by weekly EMA34 trend and volume confirmation.
In weekly uptrend (price > weekly EMA34): look for long breakouts above R1 with volume spike.
In weekly downtrend (price < weekly EMA34): look for short breakdowns below S1 with volume spike.
In ranging weekly markets (CHOP > 61.8 on 1d): fade extremes at H3/L3 levels with volume confirmation.
Uses ATR(14) stoploss (2.0x) and discrete position sizing (0.25) to manage risk and minimize fee drift.
Designed to work in both bull and bear markets by adapting to weekly trend regime.
Timeframe: 1d, uses 1w HTF for trend filter.
Target: 30-100 total trades over 4 years = 7-25/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA34 trend and chop regime)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === 1d OHLC for Camarilla pivot calculation (based on previous 1d bar) ===
    df_1d_open = prices['open'].values
    df_1d_high = prices['high'].values
    df_1d_low = prices['low'].values
    df_1d_close = prices['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    range_1d = df_1d_high - df_1d_low
    r1_1d = df_1d_close + 0.275 * range_1d
    s1_1d = df_1d_close - 0.275 * range_1d
    h3_1d = df_1d_close + 1.1 * range_1d
    l3_1d = df_1d_close - 1.1 * range_1d
    h4_1d = df_1d_close + 1.382 * range_1d
    l4_1d = df_1d_close - 1.382 * range_1d
    
    # === 1w EMA34 for trend filter ===
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Choppiness Index (14-period) on 1d for regime detection ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)  # avoid division by zero
    
    # === Volume confirmation (20-period average) ===
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
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(chop[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        chop_val = chop[i]
        r1 = r1_1d[i]
        s1 = s1_1d[i]
        h3 = h3_1d[i]
        l3 = l3_1d[i]
        h4 = h4_1d[i]
        l4 = l4_1d[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume_now > 1.5 * vol_avg
        
        if position == 0:
            # Regime-based entries with volume confirmation
            if chop_val < 38.2:  # Weekly trending regime
                # Only enter in direction of weekly trend with volume confirmation
                long_condition = (price > r1) and (price > ema_trend) and volume_confirmed
                short_condition = (price < s1) and (price < ema_trend) and volume_confirmed
                
                if long_condition:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif short_condition:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    
            elif chop_val > 61.8:  # Weekly ranging regime
                # Mean reversion at H3/L3 levels with volume confirmation
                long_condition = (price < l3) and (price > l4) and volume_confirmed  # Oversold bounce
                short_condition = (price > h3) and (price < h4) and volume_confirmed  # Overbought rejection
                
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
            # Trend reversal exit (in trending regime)
            elif chop_val < 38.2 and price < ema_trend:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit (in ranging regime)
            elif chop_val > 61.8 and price > h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (in trending regime)
            elif chop_val < 38.2 and price > ema_trend:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit (in ranging regime)
            elif chop_val > 61.8 and price < l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0