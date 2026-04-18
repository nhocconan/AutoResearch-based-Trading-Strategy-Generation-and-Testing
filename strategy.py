# NEW STRATEGY: 1h_KAMA_1dTrend_VolumeSpike_Session
# Hypothesis: KAMA adapts to market noise, reducing whipsaws in ranging markets.
# Use 1d trend filter to avoid counter-trend trades, volume spike for confirmation,
# and session filter (08-20 UTC) to reduce noise trades. Target 15-37 trades/year.

#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend direction
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # KAMA (Adaptive Moving Average) parameters
    er_period = 10
    fast_sc = 2 / (2 + 1)  # 2-period EMA smoothing constant
    slow_sc = 2 / (30 + 1) # 30-period EMA smoothing constant
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    # Handle first er_period values
    er = np.concatenate([np.full(er_period, np.nan), er])
    
    # Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[er_period] = close[er_period]  # seed
    for i in range(er_period + 1, n):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = np.nan
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume spike detection (1.5x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(100, er_period + 1)
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        above_kama = price > kama[i]
        below_kama = price < kama[i]
        above_1d_ema = price > ema_34_1d_aligned[i]
        below_1d_ema = price < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price above KAMA, above 1d EMA, volume spike, in session
            if (above_kama and above_1d_ema and volume_spike[i] and in_session[i]):
                signals[i] = 0.20
                position = 1
            # Short: price below KAMA, below 1d EMA, volume spike, in session
            elif (below_kama and below_1d_ema and volume_spike[i] and in_session[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long position: maintain 0.20 long
            signals[i] = 0.20
            # Exit: price below KAMA or below 1d EMA
            if below_kama or below_1d_ema:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: maintain -0.20 short
            signals[i] = -0.20
            # Exit: price above KAMA or above 1d EMA
            if above_kama or above_1d_ema:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_KAMA_1dTrend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0