#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h volume breakout with 4h/1d trend filter and session filter (08-20 UTC)
# Uses volume > 2.0x 20-period average for entry, filtered by 4h EMA50 and 1d EMA200.
# Trades only during 08-20 UTC to avoid low-volume sessions. Target: 60-150 total trades over 4 years (15-37/year).
# Works in bull and bear by following higher timeframe trend direction.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (trend filter) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # === 1d data (higher timeframe trend) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === Indicators ===
    # 4h EMA50 for trend
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d EMA200 for trend
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1h volume spike (2.0x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema50 = ema50_4h_aligned[i]
        ema200 = ema200_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        session_ok = in_session[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            atr_1h = np.abs(high - low)
            atr_ma = pd.Series(atr_1h).rolling(window=14, min_periods=14).mean().values
            atr_val = atr_ma[i]
            if price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_1h = np.abs(high - low)
            atr_ma = pd.Series(atr_1h).rolling(window=14, min_periods=14).mean().values
            atr_val = atr_ma[i]
            if price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 4h EMA50 or 1d EMA200
            if price < ema50 or price < ema200:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 4h EMA50 or 1d EMA200
            if price > ema50 or price > ema200:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat and in session) ===
        if position == 0 and session_ok:
            # Require volume spike and alignment with both timeframes
            if vol_spike_val:
                # Go long when price above both EMAs
                if price > ema50 and price > ema200:
                    signals[i] = 0.20
                    position = 1
                    entry_price = price
                    continue
                # Go short when price below both EMAs
                elif price < ema50 and price < ema200:
                    signals[i] = -0.20
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_VolumeBreakout_4h1dEMA_SessionFilter"
timeframe = "1h"
leverage = 1.0