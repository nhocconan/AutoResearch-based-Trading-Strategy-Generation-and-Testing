#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves; 1w EMA50 ensures alignment with weekly trend
# Volume confirmation filters false breakouts; designed for low trade frequency (7-25/year)
# Works in bull markets (breakout above upper band + 1w EMA50 uptrend) and bear markets (breakout below lower band + 1w EMA50 downtrend)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown

name = "1d_Donchian20_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50
        return np.zeros(n)
    
    # 1w EMA50 calculation
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean()
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w.values)
    
    # 1d Donchian channels (20-period)
    lookback = 20
    upper_band = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower_band = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation (20-period EMA threshold)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)  # Moderate threshold to avoid overtrading
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Donchian and EMA)
    start_idx = max(lookback, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above upper Donchian band with volume confirmation and uptrend
            if close[i] > upper_band[i] and uptrend and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below lower Donchian band with volume confirmation and downtrend
            elif close[i] < lower_band[i] and downtrend and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below lower Donchian band (reversal) OR trend changes to downtrend
            if close[i] < lower_band[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above upper Donchian band (reversal) OR trend changes to uptrend
            if close[i] > upper_band[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals