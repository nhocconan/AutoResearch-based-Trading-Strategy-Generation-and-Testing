#!/usr/bin/env python3
# 1d_Camarilla_Pivot_Squeeze_Breakout
# Hypothesis: Camarilla pivot levels on 1d provide strong support/resistance. A squeeze (low volatility) followed by breakout
# with volume confirmation captures explosive moves. Weekly trend filter ensures alignment with higher timeframe trend.
# Works in bull (breakouts up) and bear (breakouts down) markets. Target: 15-25 trades/year.

name = "1d_Camarilla_Pivot_Squeeze_Breakout"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA for trend filter (34-period)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels for 1d (using previous day's OHLC)
    # Camarilla: H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    # We use previous day's data to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # fill first value
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_h4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_l4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Volatility squeeze: ATR(5) < ATR(20) * 0.5
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    atr5 = pd.Series(tr).rolling(window=5, min_periods=5).mean().values
    atr20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: volume > 1.5 * 20-day average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for ATR and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(atr5[i]) or np.isnan(atr20[i]) or 
            np.isnan(volume_ma20[i]) or np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Squeeze condition: low volatility
        squeeze = atr5[i] < atr20[i] * 0.5
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_h4[i]
        breakout_down = close[i] < camarilla_l4[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma20[i] * 1.5
        
        # Weekly trend filter
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long entry: squeeze + breakout up + volume + weekly uptrend
            if squeeze and breakout_up and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: squeeze + breakout down + volume + weekly downtrend
            elif squeeze and breakout_down and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below Camarilla L3 or weekly trend turns down
            camarilla_l3 = prev_close[i] - (prev_high[i] - prev_low[i]) * 1.1 / 6
            if close[i] < camarilla_l3 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above Camarilla H3 or weekly trend turns up
            camarilla_h3 = prev_close[i] + (prev_high[i] - prev_low[i]) * 1.1 / 6
            if close[i] > camarilla_h3 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals