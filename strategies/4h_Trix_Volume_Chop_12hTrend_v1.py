#!/usr/bin/env python3
"""
4h_Trix_Volume_Chop_12hTrend_v1
Hypothesis: Uses TRIX (triple exponential average) momentum with volume spike confirmation and 12-hour EMA trend filter.
Trades momentum breakouts in trending markets (12h EMA50) and avoids false signals in choppy conditions using Choppiness Index.
Designed for low trade frequency (~25-35 trades/year) by requiring confluence of TRIX signal, volume spike, and trend alignment.
Works in both bull and bear markets by adapting to trend direction via 12h EMA filter.
"""

name = "4h_Trix_Volume_Chop_12hTrend_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data for trend filter and 1d data for Choppiness Index
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 14:
        return np.zeros(n)
    
    # 4h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 12h EMA50 for trend filter ---
    close_12h = df_12h['close']
    ema_50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # --- TRIX (15-period) on 4h close ---
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - 1
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100  # Percentage change
    trix = trix.fillna(0).values
    
    # --- Volume Spike Detection (2.0x 20-period EMA) ---
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ema.values)
    
    # --- Choppiness Index (14-period) on 1d data ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index = 100 * log10(tr_sum / (atr * 14)) / log10(14)
    chop = 100 * np.log10(tr_sum / (atr * 14)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)  # Replace NaN with neutral value
    
    # Align Choppiness Index to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(trix[i]) or
            np.isnan(vol_spike[i]) or
            np.isnan(chop_aligned[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend based on price vs 12h EMA50
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]
        
        # TRIX signal: positive = bullish momentum, negative = bearish momentum
        trix_bullish = trix[i] > 0.05  # Threshold to avoid noise
        trix_bearish = trix[i] < -0.05
        
        # Choppiness regime: < 38.2 = trending, > 61.8 = ranging
        trending_regime = chop_aligned[i] < 38.2
        ranging_regime = chop_aligned[i] > 61.8
        
        if position == 0:
            # In trending regime: follow TRIX momentum with volume confirmation
            if trending_regime:
                if price_above_ema and trix_bullish and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif price_below_ema and trix_bearish and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
            # In ranging regime: look for mean reversion at extreme TRIX values
            elif ranging_regime:
                if trix[i] < -0.2 and vol_spike[i]:  # Oversold
                    signals[i] = 0.25
                    position = 1
                elif trix[i] > 0.2 and vol_spike[i]:  # Overbought
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: TRIX turns negative or price breaks below 12h EMA
                exit_signal = (trix[i] < 0) or (close[i] < ema_50_12h_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: TRIX turns positive or price breaks above 12h EMA
                exit_signal = (trix[i] > 0) or (close[i] > ema_50_12h_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals