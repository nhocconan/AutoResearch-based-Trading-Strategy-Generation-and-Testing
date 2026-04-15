#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX + volume spike + chop regime filter
# Long when TRIX crosses above zero + volume > 2.0x 20-period avg + CHOP(14) < 38.2 (trending)
# Short when TRIX crosses below zero + volume > 2.0x 20-period avg + CHOP(14) < 38.2 (trending)
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# TRIX (12,20,9) provides momentum confirmation with reduced whipsaw.
# Volume threshold (2.0x) targets ~15-30 trades/year on 12h timeframe to avoid overtrading.
# CHOP regime filter ensures we only trade in trending markets, avoiding range-bound losses.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1d Indicator: EMA34 for trend filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === TRIX (12,20,9) on 12h close ===
    # TRIX = EMA(EMA(EMA(close, 12), 20), 9) - 1
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = (ema3 / np.roll(ema3, 1)) - 1  # Percentage change
    trix[0] = 0  # First value has no previous
    
    # === Choppiness Index (CHOP) on 12h ===
    # CHOP = 100 * log10(sum(ATR(1), 14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = np.maximum(high - low, np.roll(np.abs(high - np.roll(close, 1)), 1))
    tr1 = np.maximum(tr1, np.roll(np.abs(low - np.roll(close, 1)), 1))
    tr1[0] = high[0] - low[0]  # First TR
    
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr1 / (max_high - min_low)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((max_high - min_low) > 0, chop, 50.0)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 20, 14) + 5  # EMA34 + TRIX + CHOP + volume + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or np.isnan(chop[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Chop regime filter: only trade in trending markets (CHOP < 38.2)
        trending_regime = chop[i] < 38.2
        
        # === LONG CONDITIONS ===
        # 1. TRIX crosses above zero (trix[i] > 0 and trix[i-1] <= 0)
        # 2. 1d EMA34 uptrend (close > EMA34)
        # 3. Volume confirmation
        # 4. Trending regime
        if (trix[i] > 0 and trix[i-1] <= 0) and \
           (close[i] > ema_34_1d_aligned[i]) and vol_confirm and trending_regime:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. TRIX crosses below zero (trix[i] < 0 and trix[i-1] >= 0)
        # 2. 1d EMA34 downtrend (close < EMA34)
        # 3. Volume confirmation
        # 4. Trending regime
        elif (trix[i] < 0 and trix[i-1] >= 0) and \
             (close[i] < ema_34_1d_aligned[i]) and vol_confirm and trending_regime:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_TRIX_Volume_Chop_1dEMA34_v1"
timeframe = "12h"
leverage = 1.0