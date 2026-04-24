#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1d EMA50 Trend Filter + Volume Spike.
- Primary timeframe: 6h for execution.
- HTF: 1d EMA50 for trend direction (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Williams %R(14) on 6h: extreme readings (< -80 = oversold, > -20 = overbought).
- Entry: Long when Williams %R crosses above -80 AND price > 1d EMA50 (oversold bounce in uptrend).
         Short when Williams %R crosses below -20 AND price < 1d EMA50 (overbought rejection in downtrend).
- Volume confirmation: current volume > 2.0 * 20-period volume MA to avoid false signals.
- Exit: Opposite Williams %R cross (long exit at -20, short exit at -80) or EMA50 trend flip.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Why it should work: Williams %R identifies exhaustion points; EMA50 filter ensures we trade with the higher timeframe trend; volume spike confirms participation. Works in both bull (buy oversold in uptrend) and bear (sell overbought in downtrend) markets.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Williams %R(14) on 6h
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need enough bars for EMA50, volume MA, Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema_50_aligned[i]
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Check for entry signals
            if vol_spike:
                # Long: Williams %R crosses above -80 (oversold bounce) AND price > 1d EMA50 (uptrend)
                if wr > -80 and williams_r[i-1] <= -80 and close[i] > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 (overbought rejection) AND price < 1d EMA50 (downtrend)
                elif wr < -20 and williams_r[i-1] >= -20 and close[i] < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -20 (overbought) OR EMA50 trend flips down
            if wr < -20 or close[i] < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -80 (oversold) OR EMA50 trend flips up
            if wr > -80 or close[i] > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0