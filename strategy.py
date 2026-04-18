#!/usr/bin/env python3
"""
4h 12-hour EMA Trend + 4h Bollinger Breakout with Volume Confirmation
Uses 12h EMA for trend direction and 4h Bollinger Bands for entry signals.
Long when price breaks above upper band with volume and 12h EMA uptrend.
Short when price breaks below lower band with volume and 12h EMA downtrend.
Aims for low trade frequency with strong trend-following edge in both bull and bear markets.
"""

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
    
    # Get 12h data for EMA trend (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA (34 period)
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Bollinger Bands on 4h (20 period, 2 std)
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    
    # Volume spike detection (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(50, bb_period)  # need enough history
    
    for i in range(start_idx, n):
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(sma[i]) or 
            np.isnan(std[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend_up = ema_12h_aligned[i] > sma[i]  # EMA above SMA indicates uptrend
        ema_trend_down = ema_12h_aligned[i] < sma[i]  # EMA below SMA indicates downtrend
        
        if position == 0:
            # Long: price breaks above upper band with volume and 12h EMA uptrend
            if (price > upper_band[i] and 
                volume_spike[i] and ema_trend_up):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume and 12h EMA downtrend
            elif (price < lower_band[i] and 
                  volume_spike[i] and ema_trend_down):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price crosses below middle band (mean reversion)
            if price < sma[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price crosses above middle band (mean reversion)
            if price > sma[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_12hEMA34_BollingerBreakout_Volume"
timeframe = "4h"
leverage = 1.0