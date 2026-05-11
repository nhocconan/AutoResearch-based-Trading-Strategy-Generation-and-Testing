#!/usr/bin/env python3
"""
6h_Keltner_Breakout_Volume_Regime_v1
Hypothesis: Keltner Channel breakouts (ATR-based) with volume confirmation and volatility regime filter.
Uses 12h trend filter to avoid counter-trend trades. Designed for low frequency (15-30 trades/year) to work in both bull (breakouts) and bear (mean reversion near mean) markets.
Keltner Channels adapt to volatility, providing dynamic support/resistance. In high volatility regimes, breakouts are more reliable. In low volatility, we fade extremes toward the EMA middle.
"""

name = "6h_Keltner_Breakout_Volume_Regime_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 12h EMA50 for trend filter ---
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # --- Keltner Channel (20, 2.0) ---
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(high - low).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_20 + (2.0 * atr)
    lower_keltner = ema_20 - (2.0 * atr)
    
    # --- Volume spike (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)  # Volume confirmation
    
    # --- Volatility regime: ATR ratio (current vs 50-period average) ---
    atr_50 = pd.Series(high - low).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ratio = atr / atr_50  # >1 = high volatility, <1 = low volatility
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(ema_20[i]) or
            np.isnan(upper_keltner[i]) or
            np.isnan(lower_keltner[i]) or
            np.isnan(vol_spike[i]) or
            np.isnan(atr_ratio[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine volatility regime
        high_vol = atr_ratio[i] > 1.2  # High volatility regime
        low_vol = atr_ratio[i] < 0.8   # Low volatility regime
        
        # Breakout signals with volume confirmation
        breakout_long = (close[i] > upper_keltner[i-1]) and vol_spike[i]
        breakout_short = (close[i] < lower_keltner[i-1]) and vol_spike[i]
        
        # Mean reversion signals (fade extremes toward EMA middle)
        mean_revert_long = (close[i] < lower_keltner[i]) and (close[i] > ema_20[i])
        mean_revert_short = (close[i] > upper_keltner[i]) and (close[i] < ema_20[i])
        
        if position == 0:
            if high_vol:
                # High volatility: trade breakouts in trend direction
                if breakout_long and close[i] > ema_50_12h_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif breakout_short and close[i] < ema_50_12h_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif low_vol:
                # Low volatility: fade extremes toward mean
                if mean_revert_long:
                    signals[i] = 0.25
                    position = 1
                elif mean_revert_short:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Neutral volatility: no trade
                signals[i] = 0.0
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price crosses below EMA middle or opposite breakout
                exit_signal = (close[i] < ema_20[i]) or breakout_short
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above EMA middle or opposite breakout
                exit_signal = (close[i] > ema_20[i]) or breakout_long
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals