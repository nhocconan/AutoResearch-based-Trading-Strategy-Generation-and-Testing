#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume spike confirmation.
- Primary timeframe: 4h for lower trade frequency (target 20-50 trades/year per symbol).
- HTF: 1d ATR(14) for volatility regime (high ATR = trending market, low ATR = ranging).
- Volume: Current 4h volume > 1.5 * 20-period volume MA to filter low-participation moves.
- Donchian: 20-period high/low breakouts for directional entries.
- Entry: Long when price breaks above Donchian(20) high AND 1d ATR > 20-period ATR MA AND volume spike.
         Short when price breaks below Donchian(20) low AND 1d ATR > 20-period ATR MA AND volume spike.
- Exit: Time-based exit after 10 bars (approx 40 hours) or opposite Donchian breakout.
- Signal size: 0.25 discrete to minimize fee churn and control drawdown.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
This strategy captures strong trending moves confirmed by volatility expansion and volume,
avoiding false breakouts in low-volatility ranging markets. Works in both bull and bear
markets by trading breakouts in the direction of prevailing volatility regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(df_1d_high - df_1d_low)
    tr2 = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    tr3 = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_1d = np.zeros_like(tr)
    atr_1d[0] = tr[0]
    for i in range(1, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-period 1d ATR MA for regime filter
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 20-period 1d volume MA
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian(20) channels on 4h data
    # Using rolling window with min_periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 4h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_in_trade = 0  # Counter for time-based exit
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # Need enough bars for Donchian(20) and ATR MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Increment time-based exit counter
        if position != 0:
            bars_in_trade += 1
        
        if position == 0:
            # Check for breakout signals with volume spike and volatility expansion
            volume_spike = volume[i] > (1.5 * vol_ma_1d_aligned[i])
            volatility_expansion = atr_1d_aligned[i] > atr_ma_1d_aligned[i]
            
            if volume_spike and volatility_expansion:
                # Bullish breakout: price > Donchian high
                if curr_close > donchian_high[i]:
                    signals[i] = 0.25
                    position = 1
                    bars_in_trade = 0
                # Bearish breakout: price < Donchian low
                elif curr_close < donchian_low[i]:
                    signals[i] = -0.25
                    position = -1
                    bars_in_trade = 0
        elif position != 0:
            # Maintain position
            signals[i] = 0.25 if position == 1 else -0.25
            
            # Exit conditions: time-based exit (10 bars) or opposite Donchian breakout
            time_exit = bars_in_trade >= 10
            opposite_breakout = (position == 1 and curr_close < donchian_low[i]) or \
                               (position == -1 and curr_close > donchian_high[i])
            
            if time_exit or opposite_breakout:
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
    
    return signals

name = "4h_Donchian20_1dATR_Regime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0