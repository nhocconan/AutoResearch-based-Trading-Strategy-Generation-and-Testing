#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with volume confirmation and 1w ATR regime filter.
Long when price breaks above 20-day high with volume > 1.5x 20-day avg volume AND 1w ATR(14) < 30-day ATR(14) percentile 50 (low volatility regime).
Short when price breaks below 20-day low with volume > 1.5x 20-day avg volume AND 1w ATR(14) < 30-day ATR(14) percentile 50.
Exit when price touches the 10-day EMA.
Uses 1d for execution and volume, 1w for ATR regime filter.
Designed to work in both bull and bear markets by trading breakouts in low volatility regimes.
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for ATR regime filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ATR(14)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 30-day ATR(14) percentile rank (50 = median)
    atr_30d_1w = pd.Series(tr_1w).rolling(window=30*6, min_periods=30*6).mean().values  # approx 30d in 1w bars
    atr_percentile = pd.Series(atr_14_1w).rolling(window=30*6, min_periods=30*6).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, raw=False
    ).values
    low_vol_regime = atr_percentile < 50  # below median volatility
    
    # Align 1w regime to 1d timeframe
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1w, low_vol_regime)
    
    # Calculate 1d Donchian(20)
    donch_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume MA(20)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA(10) for exit
    ema_10 = pd.Series(close).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(low_vol_regime_aligned[i]) or 
            np.isnan(donch_high_20[i]) or 
            np.isnan(donch_low_20[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(ema_10[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Breakout conditions
        breakout_high = close[i] > donch_high_20[i]
        breakout_low = close[i] < donch_low_20[i]
        
        # Exit condition: touch 10-day EMA
        touch_ema = abs(close[i] - ema_10[i]) < 0.005 * close[i]
        
        if position == 0:
            # Long: break above Donchian high with volume confirmation and low vol regime
            if (breakout_high and volume_confirmed and low_vol_regime_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume confirmation and low vol regime
            elif (breakout_low and volume_confirmed and low_vol_regime_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch 10-day EMA
            if touch_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch 10-day EMA
            if touch_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Volume_LowVolATR_Regime"
timeframe = "1d"
leverage = 1.0