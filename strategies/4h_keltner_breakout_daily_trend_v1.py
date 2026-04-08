#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Keltner Channel Breakout + Daily Trend Filter + Volume Confirmation
# Hypothesis: Breakouts from Keltner Channel (ATR-based) occur with high momentum in trending markets.
# Uses daily EMA for trend filter and volume spike for confirmation. Works in both bull and bear by
# following the higher-timeframe trend. Target: 15-40 trades/year (60-160 total over 4 years).

name = "4h_keltner_breakout_daily_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    
    # Keltner Channel (20, 2.0) on 4h
    kc_period = 20
    kc_multiplier = 2.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # First bar
    tr2[0] = np.abs(high[0] - close[0])  # First bar
    tr3[0] = np.abs(low[0] - close[0])  # First bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=kc_period, min_periods=kc_period).mean().values
    
    # Keltner Channel
    ema_middle = pd.Series(close).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    upper = ema_middle + kc_multiplier * atr
    lower = ema_middle - kc_multiplier * atr
    
    # Daily EMA(50) for trend filter
    ema_50_daily = pd.Series(close_daily).ewm(span=50, adjust=False).mean().values
    ema_50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(kc_period, n):
        # Skip if required data not available
        if (np.isnan(ema_middle[i]) or np.isnan(atr[i]) or np.isnan(ema_50_daily_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price closes below middle line or trend changes to down
            if close[i] < ema_middle[i] or close[i] < ema_50_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above middle line or trend changes to up
            if close[i] > ema_middle[i] or close[i] > ema_50_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Breakout above upper channel with uptrend
                if close[i] > upper[i] and close[i] > ema_50_daily_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakdown below lower channel with downtrend
                elif close[i] < lower[i] and close[i] < ema_50_daily_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals