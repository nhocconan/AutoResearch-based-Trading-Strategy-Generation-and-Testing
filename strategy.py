#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Squeeze Breakout with 1d EMA34 trend and volume confirmation
# Bollinger Band squeeze identifies low volatility periods preceding breakouts.
# Breakout direction filtered by 1d EMA34 trend to avoid false breakouts in choppy markets.
# Volume spike confirms breakout conviction. Designed for 20-30 trades/year on 4h to minimize fee drag.
# Works in both bull and bear markets by capturing volatility expansion phases.

name = "4h_BollingerSqueeze_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Bollinger Bands (20, 2) on 4h
    bb_period = 20
    bb_std = 2
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(bb_period, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Bollinger Bands using data up to current bar
        lookback = min(bb_period, i+1)
        bb_ma = np.mean(close[i-lookback+1:i+1])
        bb_std_dev = np.std(close[i-lookback+1:i+1])
        bb_upper = bb_ma + (bb_std * bb_std_dev)
        bb_lower = bb_ma - (bb_std * bb_std_dev)
        
        # Bollinger Band Squeeze: bandwidth < 5% of middle band (low volatility)
        bandwidth = (bb_upper - bb_lower) / bb_ma if bb_ma != 0 else 0
        is_squeeze = bandwidth < 0.05
        
        # Volume confirmation: 20-period EMA
        vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1] if i >= 19 else volume[i]
        volume_spike = volume[i] > (2.0 * vol_ema_20)
        
        # Breakout conditions
        breakout_up = close[i] > bb_upper
        breakout_down = close[i] < bb_lower
        
        if position == 0:
            # Long: bullish breakout from squeeze in 1d uptrend with volume spike
            if breakout_up and is_squeeze and ema_34_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout from squeeze in 1d downtrend with volume spike
            elif breakout_down and is_squeeze and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle band or loses 1d uptrend
            if close[i] < bb_ma or ema_34_1d_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle band or loses 1d downtrend
            if close[i] > bb_ma or ema_34_1d_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals