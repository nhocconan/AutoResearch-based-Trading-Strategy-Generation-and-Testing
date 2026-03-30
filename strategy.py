#!/usr/bin/env python3
"""
Experiment #006: 4h Camarilla S3/R3 + Choppiness Regime + Volume Spike

HYPOTHESIS: Camarilla pivot levels (S3/R3) work as mean-reversion targets in
range-bound markets. Combined with Choppiness Index (regime filter) and volume
spike confirmation, this should:
- Long when price drops to S3 (oversold) + choppy range + vol spike
- Short when price rallies to R3 (overbought) + choppy range + vol spike
- Exit when choppiness drops below 38.2 (trending begins)

WHY 4h: Optimal trade frequency from DB (20-50/year). 4h captures
intraday volatility without overtrading.

KEY INSIGHT: Previous Camarilla attempt (v1) failed with Sharpe=-0.471.
Adding choppiness as regime filter should improve by:
1. Avoiding entries when market is trending (trending = Camarilla fails)
2. Only entering when market is clearly range-bound (high choppiness)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_chop_vol_v2"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - values >61.8 = choppy/range, <38.2 = trending"""
    n = len(close)
    chp = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1])) if j > 0 else high[j] - low[j]
            tr_sum += tr
        
        # Highest - Lowest over period
        hl_range = high[i - period + 1:i + 1].max() - low[i - period + 1:i + 1].min()
        
        if hl_range > 1e-10 and tr_sum > 1e-10:
            chp[i] = 100 * (np.log10(tr_sum) / np.log10(hl_range * period))
    
    return chp

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d for Camarilla pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla pivot levels from 1d data
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # R3 = Close + High-Low * 1.1/12
    cam_R3 = daily_close + (daily_high - daily_low) * (1.1 / 12)
    # R4 = Close + High-Low * 1.1/6
    cam_R4 = daily_close + (daily_high - daily_low) * (1.1 / 6)
    # S3 = Close - High-Low * 1.1/12
    cam_S3 = daily_close - (daily_high - daily_low) * (1.1 / 12)
    # S4 = Close - High-Low * 1.1/6
    cam_S4 = daily_close - (daily_high - daily_low) * (1.1 / 6)
    
    # Align to LTF
    cam_R3_aligned = align_htf_to_ltf(prices, df_1d, cam_R3)
    cam_R4_aligned = align_htf_to_ltf(prices, df_1d, cam_R4)
    cam_S3_aligned = align_htf_to_ltf(prices, df_1d, cam_S3)
    cam_S4_aligned = align_htf_to_ltf(prices, df_1d, cam_S4)
    
    # === Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chp = calculate_choppiness(high, low, close, period=14)
    
    # Volume metrics
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    warmup = 50
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(chp[i]):
            signals[i] = 0.0
            continue
        
        # Volume spike check (1.5x average)
        vol_ok = volume[i] > vol_ma20[i] * 1.5 if vol_ma20[i] > 1e-10 else False
        
        # Choppiness: > 61.8 = range-bound (Camarilla works), < 38.2 = trending (avoid)
        in_range = chp[i] > 61.8
        
        # === LONG ENTRY: Price at S3 + range-bound + volume spike ===
        if close[i] <= cam_S3_aligned[i] and in_range and vol_ok:
            signals[i] = SIZE
        
        # === SHORT ENTRY: Price at R3 + range-bound + volume spike ===
        elif close[i] >= cam_R3_aligned[i] and in_range and vol_ok:
            signals[i] = -SIZE
        
        # === EXIT CONDITIONS ===
        # Exit long: price reaches R4 or choppiness drops (trending begins)
        elif signals[i-1] > 0 and i > warmup:
            if close[i] >= cam_R4_aligned[i] or chp[i] < 38.2:
                signals[i] = 0.0
            else:
                signals[i] = SIZE
        
        # Exit short: price reaches S4 or choppiness drops
        elif signals[i-1] < 0 and i > warmup:
            if close[i] <= cam_S4_aligned[i] or chp[i] < 38.2:
                signals[i] = 0.0
            else:
                signals[i] = -SIZE
        
        else:
            signals[i] = 0.0
    
    return signals