#!/usr/bin/env python3
"""
1h_VolumeSpike_Pullback_4hTrend_v1
Hypothesis: On 1h timeframe, enter pullbacks to the 4h EMA20 during 4h trend (EMA20 > EMA50 for long, EMA20 < EMA50 for short) with volume spike confirmation (>2x 20-bar average) and session filter (08-20 UTC). Uses discrete position sizing (0.20) to limit fee drag and ATR-based stoploss (2x ATR). Designed for moderate trade frequency (15-35/year) to work in both bull (buy pullbacks in uptrend) and bear (sell rallies in downtrend) regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend filter)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h EMA20 and EMA50 for trend filter ===
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === Volume spike filter (20-period) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # === ATR for stoploss (14-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_low = prices['low'].iloc[i]
        price_high = prices['high'].iloc[i]
        ema_20_4h_val = ema_20_4h_aligned[i]
        ema_50_4h_val = ema_50_4h_aligned[i]
        vol_spike = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: 4h uptrend (EMA20 > EMA50) + pullback to EMA20 + volume spike
            if (ema_20_4h_val > ema_50_4h_val and 
                price_low <= ema_20_4h_val and 
                vol_spike > 2.0):
                signals[i] = 0.20
                position = 1
                entry_price = price_close
            # Short: 4h downtrend (EMA20 < EMA50) + rally to EMA20 + volume spike
            elif (ema_20_4h_val < ema_50_4h_val and 
                  price_high >= ema_20_4h_val and 
                  vol_spike > 2.0):
                signals[i] = -0.20
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Stoploss: 2 * ATR from entry
            if position == 1:
                if price_close < entry_price - 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if price_close > entry_price + 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_VolumeSpike_Pullback_4hTrend_v1"
timeframe = "1h"
leverage = 1.0