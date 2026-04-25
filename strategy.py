#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
Long when price breaks above 20-period Donchian high in 1d uptrend (close > 1d EMA50) with volume > 2.0x 20-bar average.
Short when price breaks below 20-period Donchian low in 1d downtrend (close < 1d EMA50) with volume > 2.0x 20-bar average.
Exit via ATR-based trailing stop (2.5*ATR from extreme) or re-entry into Donchian(10) range.
Designed for ~12-37 trades/year by requiring strong breakouts, trend alignment, and volume confirmation.
Works in bull/bear markets via 1d EMA50 filter; avoids whipsaws via volume confirmation and tight stops.
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
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR for trailing stop (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (2.0 * vol_ma_20)
    
    # Donchian channels (20-period for entry, 10-period for exit)
    donchian_len = 20
    donchian_exit_len = 10
    
    # 20-period Donchian high/low
    highest_20 = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    lowest_20 = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # 10-period Donchian high/low for exit
    highest_10 = pd.Series(high).rolling(window=donchian_exit_len, min_periods=donchian_exit_len).max().values
    lowest_10 = pd.Series(low).rolling(window=donchian_exit_len, min_periods=donchian_exit_len).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_high = 0.0   # highest high since long entry
    short_low = 0.0   # lowest low since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period, donchian_len, donchian_exit_len)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (1d EMA50 filter)
            if close[i] > ema_trend:  # 1d uptrend regime
                # Long: break above 20-period Donchian high with volume spike
                long_signal = (close[i] > highest_20[i]) and vol_regime[i]
            else:  # 1d downtrend regime
                # Short: break below 20-period Donchian low with volume spike
                short_signal = (close[i] < lowest_20[i]) and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
                long_high = high[i]
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
                short_low = low[i]
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update highest high
            if high[i] > long_high:
                long_high = high[i]
            # Exit conditions: ATR trailing stop OR re-enter Donchian(10) range
            atr_stop = long_high - 2.5 * atr[i]
            range_exit = (close[i] < highest_10[i] and close[i] > lowest_10[i])
            if close[i] <= atr_stop or range_exit:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update lowest low
            if low[i] < short_low:
                short_low = low[i]
            # Exit conditions: ATR trailing stop OR re-enter Donchian(10) range
            atr_stop = short_low + 2.5 * atr[i]
            range_exit = (close[i] > lowest_10[i] and close[i] < highest_10[i])
            if close[i] >= atr_stop or range_exit:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0