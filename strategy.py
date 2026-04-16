#!/usr/bin/env python3
"""
1D_VWAP_Bounce_Backtest
Hypothesis: On daily timeframe, price tends to revert to VWAP after extreme deviations (>1.5*ATR),
especially when aligned with weekly trend (price > weekly EMA20 for longs, < for shorts).
Volume confirmation filters out false signals. Works in both bull/bear by using VWAP as dynamic mean
and weekly EMA for trend filter. Target: 20-60 trades over 4 years.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d VWAP and ATR (primary) ===
    typical_price = (high + low + close) / 3.0
    vp = typical_price * volume
    cum_vp = np.nancumsum(vp)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_vp, cum_vol, out=np.full_like(cum_vp, np.nan), where=cum_vol!=0)
    
    # True Range for ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    def wilders_smoothing(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            result[period-1] = np.nanmean(x[1:period])
            for i in range(period, len(x)):
                result[i] = result[i-1] - (result[i-1]/period) + x[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    atr_sma = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
    
    # === 1d volume ratio for confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # === Weekly EMA20 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Align 1d indicators to 1d (no alignment needed as we're on 1d)
    # But we still align for consistency and to avoid look-ahead
    vwap_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), vwap)
    atr_sma_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), atr_sma)
    vol_ratio_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), vol_ratio)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(vwap_aligned[i]) or 
            np.isnan(atr_sma_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vwap_val = vwap_aligned[i]
        atr_val = atr_sma_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        ema_20w = ema_20_1w_aligned[i]
        
        deviation = price - vwap_val
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price returns to VWAP or breaches weekly EMA20 downward
            if abs(deviation) < 0.1 * atr_val or price < ema_20w:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price returns to VWAP or breaches weekly EMA20 upward
            if abs(deviation) < 0.1 * atr_val or price > ema_20w:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price deviates below VWAP by >1.5*ATR, with volume, and above weekly EMA20
            if deviation < -1.5 * atr_val and vol_ratio_val > 1.3 and price > ema_20w:
                signals[i] = 0.25
                position = 1
                continue
            # SHORT: Price deviates above VWAP by >1.5*ATR, with volume, and below weekly EMA20
            elif deviation > 1.5 * atr_val and vol_ratio_val > 1.3 and price < ema_20w:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1D_VWAP_Bounce_Backtest"
timeframe = "1d"
leverage = 1.0