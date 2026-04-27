#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; in strong trends, pullbacks to
# extreme levels offer high-probability entries. Combined with 1d EMA trend filter and
# volume spikes, this should work in both bull (buy oversold in uptrend) and bear
# (sell overbought in downtrend) markets. Targets 15-30 trades/year per symbol.
# Uses Williams %R(14) < -80 for long, > -20 for short, with 1d EMA50 trend and
# volume > 1.5x 20-period average.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    hh_ll = np.where(hh_ll == 0, 1e-10, hh_ll)
    williams_r = ((highest_high - close) / hh_ll) * -100
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Williams %R (14), EMA50 (50), volume MA (20)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema50_1d_aligned[i]
        wr = williams_r[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) + volume spike + uptrend (close > EMA50)
            if wr < -80 and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: Williams %R > -20 (overbought) + volume spike + downtrend (close < EMA50)
            elif wr > -20 and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R > -50 (exit oversold) or trend turns down
            if wr > -50 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Williams %R < -50 (exit overbought) or trend turns up
            if wr < -50 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsR_EMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0