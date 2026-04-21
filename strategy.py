#!/usr/bin/env python3
"""
1d_HTF_1w_Camarilla_R1S1_Breakout_VolumeATRFilter_V1
Hypothesis: Daily Camarilla R1/S1 breakouts with weekly trend filter (price > weekly EMA34 for longs, < for shorts) and volume confirmation (>1.5x 20-period volume MA). 
Weekly EMA34 provides robust trend regime for BTC/ETH in both bull and bear markets. 
Volume confirmation reduces false breakouts. ATR-based stoploss manages risk. 
Target 7-25 trades/year (30-100 total over 4 years) on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === 1w EMA34 for trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily Camarilla pivot levels (R1, S1) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate daily Camarilla levels using prior day's OHLC
    # Shift by 1 to use prior day's data (no look-ahead)
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close = np.roll(close, 1)
    prior_high[0] = prior_low[0] = prior_close[0] = np.nan  # first bar has no prior
    
    pivot = (prior_high + prior_low + prior_close) / 3
    range_hl = prior_high - prior_low
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)
    
    # === Volume confirmation (20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR for stoploss (14-period) ===
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_multiplier = 2.5  # ATR stoploss multiplier
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + weekly uptrend
            if price > r1[i] and vol_ok and price > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 + volume confirmation + weekly downtrend
            elif price < s1[i] and vol_ok and price < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position management
            # Stoploss: price drops below entry_price - atr_multiplier * atr
            if price < entry_price - atr_multiplier * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks below weekly EMA34 (trend change)
            elif price < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short position management
            # Stoploss: price rises above entry_price + atr_multiplier * atr
            if price > entry_price + atr_multiplier * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks above weekly EMA34 (trend change)
            elif price > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_HTF_1w_Camarilla_R1S1_Breakout_VolumeATRFilter_V1"
timeframe = "1d"
leverage = 1.0