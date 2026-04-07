#!/usr/bin/env python3
"""
1d_volatility_breakout_1w_trend_v1
Hypothesis: On daily timeframe, use volatility contraction breakouts with weekly trend filter.
Long when price breaks above Bollinger upper band during weekly uptrend with volume confirmation.
Short when price breaks below Bollinger lower band during weekly downtrend with volume confirmation.
Exit when price returns to Bollinger middle band.
Designed for 15-25 trades/year to minimize fee decay while capturing explosive moves in both bull and bear markets.
Volatility contraction precedes major moves; weekly trend filter avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_volatility_breakout_1w_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Determine weekly trend direction (using EMA slope)
    weekly_trend_up = np.zeros(len(ema_20_1w_aligned), dtype=bool)
    weekly_trend_down = np.zeros(len(ema_20_1w_aligned), dtype=bool)
    for i in range(1, len(ema_20_1w_aligned)):
        if not np.isnan(ema_20_1w_aligned[i]) and not np.isnan(ema_20_1w_aligned[i-1]):
            weekly_trend_up[i] = ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]
            weekly_trend_down[i] = ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]
    
    # Bollinger Bands (20, 2) on daily timeframe
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = sma_20 + (bb_std * std_20)
    bb_lower = sma_20 - (bb_std * std_20)
    bb_middle = sma_20
    
    # Volume filter: 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(20, 50), n):
        # Skip if data not available
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price returns to Bollinger middle band
            if close[i] <= bb_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Bollinger middle band
            if close[i] >= bb_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation and weekly trend alignment
            if vol_ok:
                # Long: price breaks above BB upper with weekly uptrend
                if (close[i] > bb_upper[i] and close[i-1] <= bb_upper[i-1] and 
                    weekly_trend_up[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below BB lower with weekly downtrend
                elif (close[i] < bb_lower[i] and close[i-1] >= bb_lower[i-1] and 
                      weekly_trend_down[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals