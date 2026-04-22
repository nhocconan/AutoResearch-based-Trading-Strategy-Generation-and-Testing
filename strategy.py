#!/usr/bin/env python3
"""
Hypothesis: 4-hour Bollinger Band Squeeze with 1-day ATR volatility regime filter and volume confirmation.
Trades breakouts from low volatility periods (Bollinger Band width < 20th percentile) in the direction of
the daily ATR trend (increasing volatility = trend continuation). Uses volume spike to confirm breakout
strength. Designed for low trade frequency (15-30 trades/year) to minimize fee flood and work in both
bull and bear markets by trading volatility expansions rather than directional bias.
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
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * std
    lower = sma - bb_std * std
    bb_width = upper - lower
    
    # Daily ATR for volatility regime (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Bollinger Band width percentile (20-period lookback)
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(bb_width_percentile[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(upper[i]) or np.isnan(lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: low volatility (BB width < 30th percentile) AND increasing volatility (ATR rising)
        low_vol_regime = bb_width_percentile[i] < 0.30
        vol_increasing = atr_14_1d_aligned[i] > atr_14_1d_aligned[i-1] if i > 0 else False
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0 and low_vol_regime and vol_increasing and vol_spike:
            # Breakout direction: above upper band = long, below lower band = short
            if close[i] > upper[i]:
                signals[i] = 0.25
                position = 1
            elif close[i] < lower[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: volatility contraction (BB width expanding) or opposite band touch
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches lower band or volatility contracting
                if close[i] < lower[i] or bb_width_percentile[i] > 0.70:
                    exit_signal = True
            elif position == -1:
                # Exit short: price touches upper band or volatility contracting
                if close[i] > upper[i] or bb_width_percentile[i] > 0.70:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_BollingerSqueeze_ATRVolatilityRegime_Volume"
timeframe = "4h"
leverage = 1.0