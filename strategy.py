#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily Donchian(20) breakout with 1-week EMA50 trend filter and volume confirmation.
Long when price breaks above 20-day high in 1w uptrend (close > 1w EMA50) with volume > 2.5x 20-day average.
Short when price breaks below 20-day low in 1w downtrend (close < 1w EMA50) with volume > 2.5x 20-day average.
Exit via Donchian(10) opposite break or ATR trailing stop (2.5*ATR from extreme).
Designed for ~15-25 trades/year by requiring strong volume spike and clear trend alignment.
Works in bull/bear markets via 1w EMA50 filter; avoids whipsaws via volume confirmation.
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
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period for entry, 10-period for exit)
    lookback_high = 20
    lookback_low = 20
    exit_lookback = 10
    
    # Donchian high (20-period)
    donch_high = pd.Series(high).rolling(window=lookback_high, min_periods=lookback_high).max().values
    # Donchian low (20-period)
    donch_low = pd.Series(low).rolling(window=lookback_low, min_periods=lookback_low).min().values
    # Donchian high for exit (10-period)
    donch_high_exit = pd.Series(high).rolling(window=exit_lookback, min_periods=exit_lookback).max().values
    # Donchian low for exit (10-period)
    donch_low_exit = pd.Series(low).rolling(window=exit_lookback, min_periods=exit_lookback).min().values
    
    # Volume regime: volume > 2.5x 20-day average (stricter for fewer trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (2.5 * vol_ma_20)
    
    # ATR for trailing stop (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(50, lookback_high, atr_period)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(donch_high_exit[i]) or np.isnan(donch_low_exit[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1w_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (1w EMA50 filter)
            if close[i] > ema_trend:  # 1w uptrend regime
                # Long: break above 20-day Donchian high with volume spike
                long_signal = (high[i] > donch_high[i]) and vol_regime[i]
            else:  # 1w downtrend regime
                # Short: break below 20-day Donchian low with volume spike
                short_signal = (low[i] < donch_low[i]) and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update extreme (highest high)
            if high[i] > long_extreme:
                long_extreme = high[i]
            # Exit conditions: Donchian(10) low break OR ATR trailing stop
            donch_exit = low[i] < donch_low_exit[i]
            atr_stop = long_extreme - 2.5 * atr[i]
            if donch_exit or close[i] <= atr_stop:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update extreme (lowest low)
            if low[i] < short_extreme:
                short_extreme = low[i]
            # Exit conditions: Donchian(10) high break OR ATR trailing stop
            donch_exit = high[i] > donch_high_exit[i]
            atr_stop = short_extreme + 2.5 * atr[i]
            if donch_exit or close[i] >= atr_stop:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0