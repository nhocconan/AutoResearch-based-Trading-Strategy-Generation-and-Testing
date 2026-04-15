#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Williams %R extreme + volume spike + weekly trend filter
# Long when Williams %R(14) < -80 (oversold) + volume > 2.0x 20-day avg + price > weekly EMA34 (bullish weekly trend)
# Short when Williams %R(14) > -20 (overbought) + volume > 2.0x 20-day avg + price < weekly EMA34 (bearish weekly trend)
# Uses 1d timeframe for signal generation, 1w for trend filter
# Designed for low trade frequency (<25/year) to minimize fee drag in ranging/bear markets (2025+ test)
# Williams %R captures mean reversion extremes; volume confirms conviction; weekly EMA avoids counter-trend trades

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: Williams %R (14) ===
    highest_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # === 1d Indicator: Volume SMA (20) for confirmation ===
    vol_sma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20)
    
    # === 1w HTF: EMA34 (trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_sma_20_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = df_1d['volume'].iloc[i] > (vol_sma_20_aligned[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Williams %R < -80 (oversold)
        # 2. Above weekly EMA34 (bullish weekly trend)
        # 3. Volume confirmation
        if (williams_r_aligned[i] < -80) and \
           (close[i] > ema_34_1w_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R > -20 (overbought)
        # 2. Below weekly EMA34 (bearish weekly trend)
        # 3. Volume confirmation
        elif (williams_r_aligned[i] > -20) and \
             (close[i] < ema_34_1w_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_WilliamsR_Volume_WeeklyEMA34_v1"
timeframe = "1d"
leverage = 1.0