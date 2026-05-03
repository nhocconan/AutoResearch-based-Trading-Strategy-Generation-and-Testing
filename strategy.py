#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Squeeze Breakout with 1d EMA34 trend filter and volume confirmation
# Bollinger Band squeeze (low volatility) precedes breakouts. Breakout in direction of 1d EMA34 trend
# with volume confirmation captures explosive moves. Designed for 20-40 trades/year on 4h to minimize fee drag.
# Works in both bull and bear markets by trading breakouts in the direction of the higher timeframe trend.

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
    
    # Calculate Bollinger Bands (20, 2)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient warmup for Bollinger Bands
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Bollinger Bands using data up to current bar
        lookback = min(20, i+1)
        ma = np.mean(close[i-lookback+1:i+1])
        std = np.std(close[i-lookback+1:i+1])
        upper_band = ma + 2 * std
        lower_band = ma - 2 * std
        
        # Bollinger Band Squeeze: Bandwidth < 5% of MA (low volatility)
        bandwidth = (upper_band - lower_band) / ma if ma != 0 else 0
        squeeze = bandwidth < 0.05
        
        # Volume confirmation: 20-period EMA
        vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1] if i >= 19 else volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        # Breakout conditions
        breakout_up = close[i] > upper_band
        breakout_down = close[i] < lower_band
        
        if position == 0:
            # Long: bullish breakout during squeeze in 1d uptrend with volume spike
            if squeeze and breakout_up and ema_34_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout during squeeze in 1d downtrend with volume spike
            elif squeeze and breakout_down and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to mean or loses 1d uptrend
            if close[i] < ma or ema_34_1d_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to mean or loses 1d downtrend
            if close[i] > ma or ema_34_1d_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals