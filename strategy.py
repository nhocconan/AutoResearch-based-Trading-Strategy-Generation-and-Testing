# 1h_MultiTF_Breakout_Momentum
# Hypothesis: Combines 4h Donchian breakout with 1d momentum filter and volume confirmation.
# Uses 4h for trend direction, 1d for regime filter, and 1h for precise entry timing.
# Designed for low trade frequency (15-25/year) to avoid fee drag, works in both bull and bear markets.
# Volume spike and momentum filter reduce false breakouts.

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
    
    # === 4H DONCHIAN CHANNEL (20-period) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian channels: 20-period high/low
    donch_high = np.full(len(high_4h), np.nan)
    donch_low = np.full(len(low_4h), np.nan)
    for i in range(20, len(high_4h)):
        donch_high[i] = np.max(high_4h[i-20:i])
        donch_low[i] = np.min(low_4h[i-20:i])
    
    # Breakout signals
    breakout_up = high_4h > donch_high  # New 20-period high
    breakout_down = low_4h < donch_low   # New 20-period low
    
    # Align to 1h timeframe (wait for 4h bar to close)
    breakout_up_1h = align_htf_to_ltf(prices, df_4h, breakout_up.astype(float))
    breakout_down_1h = align_htf_to_ltf(prices, df_4h, breakout_down.astype(float))
    
    # === 1D MOMENTUM FILTER (RSI > 50 for uptrend, < 50 for downtrend) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # RSI(14) on daily
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:14])
            avg_loss[i] = np.mean(loss[0:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.full(len(close_1d), np.nan)
    rsi_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi_1d[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi_1d[i] = 100
    
    # Momentum filter: RSI > 50 = bullish regime, RSI < 50 = bearish regime
    bullish_regime = rsi_1d > 50
    bearish_regime = rsi_1d < 50
    
    # Align to 1h
    bullish_regime_1h = align_htf_to_ltf(prices, df_1d, bullish_regime.astype(float))
    bearish_regime_1h = align_htf_to_ltf(prices, df_1d, bearish_regime.astype(float))
    
    # === 1H VOLUME CONFIRMATION ===
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)  # 50% above average
    
    # === SESSION FILTER (08-20 UTC) ===
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(breakout_up_1h[i]) or np.isnan(breakout_down_1h[i]) or 
            np.isnan(bullish_regime_1h[i]) or np.isnan(bearish_regime_1h[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 4h bullish breakout + 1d bullish regime + volume spike
            if breakout_up_1h[i] > 0.5 and bullish_regime_1h[i] > 0.5 and vol_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: 4h bearish breakout + 1d bearish regime + volume spike
            elif breakout_down_1h[i] > 0.5 and bearish_regime_1h[i] > 0.5 and vol_spike[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: 4h bearish breakout or 1d turns bearish
            if breakout_down_1h[i] > 0.5 or bearish_regime_1h[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: 4h bullish breakout or 1d turns bullish
            if breakout_up_1h[i] > 0.5 or bullish_regime_1h[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_MultiTF_Breakout_Momentum"
timeframe = "1h"
leverage = 1.0