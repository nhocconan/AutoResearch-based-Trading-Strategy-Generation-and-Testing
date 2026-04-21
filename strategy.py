#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_VolumeSpike_ChopRegime_v1
Hypothesis: On daily timeframe, Camarilla pivot levels (R3/S3) act as institutional support/resistance. 
Enter long at S3 with volume spike in choppy market (CHOP > 61.8), short at R3 with volume spike in choppy market.
Use 1w EMA(34) as trend filter: only long when price > weekly EMA, short when price < weekly EMA.
ATR-based stoploss (2.5x ATR) and time-based exit (10 days max hold) to control drawdown.
Designed to work in both bull (mean reversion at extremes) and bear (fades at resistance/support).
Target: 20-60 trades over 4 years (~5-15/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load weekly HTF data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === Daily Camarilla Pivot Levels (from previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels using previous day's OHLC
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    r3 = pivot + range_hl * 1.1 / 2
    s3 = pivot - range_hl * 1.1 / 2
    
    # Align to 1d timeframe (use previous day's levels for today)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === Weekly EMA(34) for trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily indicators for entry timing ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: volume > 2.0 * 20-day average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Choppiness Index regime filter: CHOP > 61.8 = ranging market (good for mean reversion)
    high = prices['high'].values
    low = prices['low'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_14 - ll_14
    # Avoid division by zero
    chop = np.where(range_14 > 0, 100 * np.log10(sum_tr_14 / range_14) / np.log10(14), 50)
    chop_regime = chop > 61.8  # ranging/choppy market
    
    # === ATR for stoploss and position sizing ===
    atr = atr_14  # use 14-period ATR
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = -1  # track entry bar for time-based exit
    entry_price = 0.0  # track entry price for stoploss
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop_regime[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_bar = -1
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        is_chop = chop_regime[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_trend = ema_34_1w_aligned[i]
        atr_val = atr[i]
        
        # Update trailing stoploss and check time-based exit
        if position != 0:
            # Time-based exit: max 10 days hold
            if i - entry_bar >= 10:
                signals[i] = 0.0
                position = 0
                entry_bar = -1
                continue
            
            # ATR-based stoploss
            if position == 1:  # long
                stop_price = entry_price - 2.5 * atr_val
                if price <= stop_price:
                    signals[i] = 0.0
                    position = 0
                    entry_bar = -1
                    continue
            else:  # short
                stop_price = entry_price + 2.5 * atr_val
                if price >= stop_price:
                    signals[i] = 0.0
                    position = 0
                    entry_bar = -1
                    continue
        
        # Entry logic
        if position == 0:
            # Long: price at S3 + volume spike + choppy market + above weekly EMA
            if (price <= s3_val and 
                vol_spike and 
                is_chop and 
                price > ema_trend):
                signals[i] = 0.25
                position = 1
                entry_bar = i
                entry_price = price
            # Short: price at R3 + volume spike + choppy market + below weekly EMA
            elif (price >= r3_val and 
                  vol_spike and 
                  is_chop and 
                  price < ema_trend):
                signals[i] = -0.25
                position = -1
                entry_bar = i
                entry_price = price
        
        elif position != 0:
            # Hold position
            signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Camarilla_Pivot_VolumeSpike_ChopRegime_v1"
timeframe = "1d"
leverage = 1.0