#!/usr/bin/env python3
"""
12h_WilliamsAlligator_MeanReversion_v1
Hypothesis: Williams Alligator (13,8,5 SMAs) identifies trend phases; price crossing outside Alligator mouth
with RSI(14) mean reversion on 1d (RSI<30 long, RSI>70 short) and volume confirmation provides
high-probability mean-reversion trades in ranging markets. Works in both bull/bear via regime filter:
ADX(14) < 25 on 1d (low trend strength) to avoid trending markets where mean reversion fails.
Target: 15-25 trades/year on 12h timeframe to minimize fee drag.
"""

name = "12h_WilliamsAlligator_MeanReversion_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for HTF indicators (ADX, RSI)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Williams Alligator on 12h (13,8,5 SMAs) ---
    # Jaw (13-period), Teeth (8-period), Lips (5-period) - all SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    
    # --- 1d ADX (14-period) for regime filter ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and Directional Movement
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum()
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum()
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / (tr14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr14 + 1e-10)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    adx_values = adx.values
    
    # --- 1d RSI (14-period) ---
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    
    # Align HTF indicators to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # --- Volume Spike (12h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (max of Alligator jaw and HTF indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if HTF data not ready
        if np.isnan(adx_aligned[i]) or np.isnan(rsi_1d_aligned[i]):
            if position != 0:
                # Exit if Alligator starts to converge (teeth crosses lips) or ADX rises
                if position == 1 and teeth[i] <= lips[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and teeth[i] >= lips[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Regime filter: only trade in low ADX (ranging market)
        ranging_market = adx_aligned[i] < 25
        
        # Mean reentry signals: price outside Alligator mouth with RSI extreme
        price_above_teeth = close[i] > teeth[i]
        price_below_teeth = close[i] < teeth[i]
        rsi_oversold = rsi_1d_aligned[i] < 30
        rsi_overbought = rsi_1d_aligned[i] > 70
        
        long_entry = ranging_market and price_below_teeth and rsi_oversold and vol_spike[i]
        short_entry = ranging_market and price_above_teeth and rsi_overbought and vol_spike[i]
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price re-enters Alligator mouth or regime changes
            if position == 1:
                if (close[i] >= teeth[i]) or (adx_aligned[i] >= 25) or (rsi_1d_aligned[i] >= 70):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if (close[i] <= teeth[i]) or (adx_aligned[i] >= 25) or (rsi_1d_aligned[i] <= 30):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals