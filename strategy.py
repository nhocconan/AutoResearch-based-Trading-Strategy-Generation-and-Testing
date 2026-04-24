#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ATR(14) for volatility regime (high volatility = trending market).
- Donchian channels: 20-period high/low breakouts for trend following.
- Entry: Long when price breaks above 20-period Donchian high AND 1d ATR > 1.5 * ATR MA(50) AND volume > 1.5 * volume MA(20).
         Short when price breaks below 20-period Donchian low AND 1d ATR > 1.5 * ATR MA(50) AND volume > 1.5 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below 20-period Donchian mid,
        exit short when price crosses above 20-period Donchian mid.
- Signal size: 0.25 discrete to balance return and drawdown.
Uses volatility regime filter to avoid whipsaws in low-volatility ranging markets and capture strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range calculation
    tr1 = df_1d_high - df_1d_low
    tr2 = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    tr3 = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr1[0] = df_1d_high[0] - df_1d_low[0]  # First period
    tr2[0] = np.abs(df_1d_high[0] - df_1d_close[0])
    tr3[0] = np.abs(df_1d_low[0] - df_1d_close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # ATR MA(50) for regime filter
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 20-period Donchian channels on 12h data
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_m = (donchian_h + donchian_l) / 2.0  # Midline for exit
    
    # Align HTF indicators to 12h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    # Calculate volume MA(20) for confirmation (using 12h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 60, 50)  # Need enough bars for Donchian, ATR, and ATR MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or np.isnan(donchian_m[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volatility regime and volume confirmation
            vol_regime = atr_1d_aligned[i] > 1.5 * atr_ma_50_aligned[i]  # High volatility = trending
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Long: Price breaks above Donchian high AND volatility regime AND volume confirmed
            if curr_close > donchian_h[i] and vol_regime and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND volatility regime AND volume confirmed
            elif curr_close < donchian_l[i] and vol_regime and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below Donchian midline (trend weakening)
            if curr_close < donchian_m[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above Donchian midline (trend weakening)
            if curr_close > donchian_m[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATR_Regime_VolumeConfirmation_v1"
timeframe = "12h"
leverage = 1.0