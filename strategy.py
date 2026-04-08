#!/usr/bin/env python3
"""
6H Bollinger Band Width + Daily Trend + Volume Spike
Hypothesis: Bollinger Band Width (BBW) identifies volatility regimes. Low BBW (<20th percentile) indicates compression/squeeze.
Breakouts from low volatility with daily EMA trend alignment and volume spike capture explosive moves in both bull and bear markets.
Designed for 6h timeframe to capture multi-day swings with controlled trade frequency (target: 15-30 trades/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_bb_width_daily_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * bb_std_dev
    lower = sma - bb_std * bb_std_dev
    bb_width = (upper - lower) / sma  # Normalized width
    
    # BBW percentile lookback (50 periods ~ 12.5 days)
    bb_width_series = pd.Series(bb_width)
    bb_width_pct = bb_width_series.rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_21 = df_1d['close'].ewm(span=21, adjust=False).mean().values
    ema_21_6h = align_htf_to_ltf(prices, df_1d, ema_21)
    
    # Volume spike (>2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(bb_period, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width_pct[i]) or np.isnan(ema_21_6h[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: BBW expands significantly (breakout fading) or trend reverses
            if bb_width_pct[i] > 80 or close[i] < ema_21_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: BBW expands significantly or trend reverses
            if bb_width_pct[i] > 80 or close[i] > ema_21_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Entry: Low volatility squeeze (BBW < 20th percentile) + breakout + volume spike + trend alignment
            if bb_width_pct[i] < 20:
                # Breakout above upper band with volume and trend
                if (close[i] > upper[i] and 
                    close[i] > ema_21_6h[i] and 
                    vol_spike[i]):
                    position = 1
                    signals[i] = 0.25
                # Breakout below lower band with volume and trend
                elif (close[i] < lower[i] and 
                      close[i] < ema_21_6h[i] and 
                      vol_spike[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals