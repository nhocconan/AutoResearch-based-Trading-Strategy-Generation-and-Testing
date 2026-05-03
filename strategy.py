#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d EMA34 trend and volume confirmation
# Bollinger Band squeeze (low volatility) precedes explosive moves. Breakout in direction of 1d EMA34 trend
# captures trending moves after consolidation. Volume spike confirms conviction. Designed for 20-40 trades/year
# on 4h to minimize fee drag. Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend).

name = "4h_BB_Squeeze_Breakout_1dEMA34_Volume"
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
    
    # Calculate Bollinger Bands (20, 2) on 4h
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient warmup for BB
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
        bb_width = (upper_band - lower_band) / ma if ma != 0 else 0
        
        # Bollinger Band squeeze: width below 20-period average width
        if i >= 39:  # Need 20+20 for width average
            width_lookback = min(20, i+1)
            width_sum = 0
            width_count = 0
            for j in range(max(0, i-19), i+1):
                if j >= 20:
                    lookback_j = min(20, j+1)
                    ma_j = np.mean(close[j-lookback_j+1:j+1])
                    std_j = np.std(close[j-lookback_j+1:j+1])
                    if ma_j != 0:
                        width_j = ((ma_j + 2 * std_j) - (ma_j - 2 * std_j)) / ma_j
                        width_sum += width_j
                        width_count += 1
            avg_width = width_sum / width_count if width_count > 0 else 0
            squeeze = bb_width < avg_width * 0.5  # Squeeze threshold
        else:
            squeeze = False
        
        # Breakout conditions
        breakout_up = close[i] > upper_band
        breakout_down = close[i] < lower_band
        
        # Volume confirmation: 20-period EMA
        if i >= 19:
            vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            vol_ema_20 = volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        if position == 0:
            # Long: bullish breakout from squeeze in 1d uptrend with volume spike
            if squeeze and breakout_up and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakdown from squeeze in 1d downtrend with volume spike
            elif squeeze and breakout_down and ema_34_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle band or loses 1d uptrend
            if close[i] < ma or ema_34_1d_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle band or loses 1d downtrend
            if close[i] > ma or ema_34_1d_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals