#!/usr/bin/env python3
"""
12h Bullish/Bearish Engulfing with 1w EMA Trend and Volume Confirmation
Hypothesis: Engulfing candles on 12h timeframe signal strong momentum reversals.
            Filter by weekly EMA200 to ensure trades align with long-term trend.
            Volume confirmation ensures institutional participation.
            Designed for low-frequency, high-conviction trades in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly EMA to 12h timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 200  # need enough history for EMA and pattern
    
    for i in range(start_idx, n):
        if (np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Bullish engulfing: current green candle fully engulfs previous red candle
        bullish_engulf = (close[i] > open_price[i] and  # current green
                          close[i-1] < open_price[i-1] and  # previous red
                          close[i] >= open_price[i-1] and   # current close >= prev open
                          open_price[i] <= close[i-1])      # current open <= prev close
        
        # Bearish engulfing: current red candle fully engulfs previous green candle
        bearish_engulf = (close[i] < open_price[i] and  # current red
                          close[i-1] > open_price[i-1] and  # previous green
                          close[i] <= open_price[i-1] and   # current close <= prev open
                          open_price[i] >= close[i-1])      # current open >= prev close
        
        ema_200 = ema_200_1w_aligned[i]
        
        if position == 0:
            # Long: bullish engulfing above weekly EMA200 with volume spike
            if bullish_engulf and close[i] > ema_200 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing below weekly EMA200 with volume spike
            elif bearish_engulf and close[i] < ema_200 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: bearish engulfing or price crosses below weekly EMA200
            if bearish_engulf or close[i] < ema_200:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: bullish engulfing or price crosses above weekly EMA200
            if bullish_engulf or close[i] > ema_200:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Engulfing_Trend_Volume"
timeframe = "12h"
leverage = 1.0