#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and 1w ATR regime filter.
- Primary timeframe: 12h for execution, HTF: 1d for Donchian channels, 1w for ATR-based volatility regime.
- ATR regime filter: High volatility (ATR > 20-period ATR MA) favors breakout strategies; low volatility favors mean reversion.
- Entry: Long when price breaks above 20-period Donchian high AND volume spike AND high volatility regime.
         Short when price breaks below 20-period Donchian low AND volume spike AND high volatility regime.
         In low volatility regime: Long when price touches Donchian low AND reverses up; Short when touches high AND reverses down.
- Exit: Opposite Donchian breakout or volatility regime shift.
- Volume confirmation: current volume > 2.0 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 1d
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Get 1w data for ATR regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate ATR (14-period) on 1w
    high_1w = pd.Series(df_1w['high'])
    low_1w = pd.Series(df_1w['low'])
    close_1w = pd.Series(df_1w['close'])
    
    tr1 = (high_1w - low_1w).abs()
    tr2 = (high_1w - close_1w.shift()).abs()
    tr3 = (low_1w - close_1w.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR moving average for regime filter
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    high_volatility = atr > atr_ma  # True when ATR > MA (high volatility regime)
    
    # Align HTF indicators to 12h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    high_volatility_aligned = align_htf_to_ltf(prices, df_1w, high_volatility)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough 1w bars for ATR MA and 20 for Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(high_volatility_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        is_high_vol = high_volatility_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        upper_channel = donchian_high_aligned[i]
        lower_channel = donchian_low_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if is_high_vol:  # High volatility regime: breakout strategy
                    # Bullish breakout: price closes above upper channel
                    if curr_close > upper_channel:
                        signals[i] = 0.25
                        position = 1
                    # Bearish breakout: price closes below lower channel
                    elif curr_close < lower_channel:
                        signals[i] = -0.25
                        position = -1
                else:  # Low volatility regime: mean reversion at extremes
                    # Long when price touches lower channel and shows reversal (close > low)
                    if curr_low <= lower_channel and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches upper channel and shows reversal (close < high)
                    elif curr_high >= upper_channel and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price closes below lower channel OR volatility regime shifts to low
            if curr_close < lower_channel or not is_high_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above upper channel OR volatility regime shifts to low
            if curr_close > upper_channel or not is_high_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_1wATRRegime_v1"
timeframe = "12h"
leverage = 1.0