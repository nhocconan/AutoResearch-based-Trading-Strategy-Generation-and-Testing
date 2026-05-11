#!/usr/bin/env python3
"""
4h_Keltner_Breakout_Volume_Regime_v1
Hypothesis: Keltner Channel breakouts with volume confirmation and ADX regime filter.
In trending markets (ADX>25), trade breakouts in direction of trend.
In ranging markets (ADX<20), avoid trades to prevent whipsaw.
Designed for low trade frequency (<30/year) to avoid fee decay while capturing strong moves.
"""

name = "4h_Keltner_Breakout_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 4h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d ADX for regime filter (14-period) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    dm_plus14 = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
    dm_minus14 = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    adx[np.isnan(dx)] = 0
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # --- Keltner Channel (20-period EMA, 2*ATR) ---
    # EMA of typical price
    typical_price = (high + low + close) / 3
    ema_tp = pd.Series(typical_price).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR
    tr_4h = np.maximum(high - low,
                       np.maximum(np.abs(high - np.roll(close, 1)),
                                  np.abs(low - np.roll(close, 1))))
    tr_4h[0] = high[0] - low[0]
    atr = pd.Series(tr_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Bands
    upper_keltner = ema_tp + 2 * atr
    lower_keltner = ema_tp - 2 * atr
    
    # --- Volume Spike (2x 20-period EMA) ---
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ema)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(ema_tp[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Regime filter
        adx_val = adx_aligned[i]
        trending = adx_val > 25
        ranging = adx_val < 20
        
        # Breakout conditions
        bullish_breakout = high[i] > upper_keltner[i] and vol_spike[i]
        bearish_breakout = low[i] < lower_keltner[i] and vol_spike[i]
        
        if position == 0:
            if trending:
                # In trending markets, trade breakouts in direction of trend
                if bullish_breakout and close[i] > ema_tp[i]:
                    signals[i] = 0.25
                    position = 1
                elif bearish_breakout and close[i] < ema_tp[i]:
                    signals[i] = -0.25
                    position = -1
            # In ranging or transitional markets, avoid new entries to prevent whipsaw
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to middle or opposite band touch
                if close[i] < ema_tp[i] or low[i] < lower_keltner[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to middle or opposite band touch
                if close[i] > ema_tp[i] or high[i] > upper_keltner[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals