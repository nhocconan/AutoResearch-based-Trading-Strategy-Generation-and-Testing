#!/usr/bin/env python3
# 1d_ema_trend_volume_v1
# Hypothesis: Daily strategy using EMA(50) trend filter with volume confirmation (>1.3x 20-period average) and 1w HTF trend alignment (price > 20-period EMA). Enters long when price is above EMA(50) with volume confirmation and bullish 1w trend; short when price is below EMA(50) with volume confirmation and bearish 1w trend. Uses discrete position sizing (0.25) to limit fee drag. Target: 7-25 trades/year to work in both bull and bear markets by following volume-confirmed trends aligned with higher timeframe direction.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # EMA(50) on primary timeframe (1d)
    close_s = pd.Series(close)
    ema_50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1w HTF trend filter: 20-period EMA on 1w timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(ema_50[i]) or
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA(50)
            if close[i] < ema_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above EMA(50)
            if close[i] > ema_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with volume confirmation and 1w trend alignment
            if volume_confirmed:
                # Bullish 1w trend: price above 20-period EMA
                bullish_trend = close[i] > ema_20_1w_aligned[i]
                # Bearish 1w trend: price below 20-period EMA
                bearish_trend = close[i] < ema_20_1w_aligned[i]
                
                # Long: price above EMA(50) with volume and bullish 1w trend
                if close[i] > ema_50[i] and bullish_trend:
                    position = 1
                    signals[i] = 0.25
                # Short: price below EMA(50) with volume and bearish 1w trend
                elif close[i] < ema_50[i] and bearish_trend:
                    position = -1
                    signals[i] = -0.25
    
    return signals

name = "1d_ema_trend_volume_v1"
timeframe = "1d"
leverage = 1.0