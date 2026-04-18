#!/usr/bin/env python3
"""
6h Bollinger Squeeze Breakout with Volume and 12h Trend Filter
Uses Bollinger Band width contraction (squeeze) followed by expansion with volume confirmation.
Breakout direction determined by 12h EMA trend filter. Designed for low trade frequency
with clear entry/exit rules to minimize whipsaw in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + (bb_std * std)
    lower = sma - (bb_std * std)
    bb_width = (upper - lower) / sma  # Normalized width
    
    # Bollinger Squeeze detection: width below 20-period average
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Breakout detection: price outside bands after squeeze
    breakout_up = (close > upper) & squeeze
    breakout_down = (close < lower) & squeeze
    
    # Volume confirmation: 1.5x 4-period average
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(bb_width_ma[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_34_12h_aligned[i]
        
        if position == 0:
            # Long: bullish breakout above upper BB with volume and above 12h EMA
            if (breakout_up[i] and 
                volume_confirmed[i] and 
                price > ema_trend):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: bearish breakout below lower BB with volume and below 12h EMA
            elif (breakout_down[i] and 
                  volume_confirmed[i] and 
                  price < ema_trend):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position: hold until reversal below 12h EMA
            signals[i] = 0.25
            if price < ema_trend:  # Trend reversal
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: hold until reversal above 12h EMA
            signals[i] = -0.25
            if price > ema_trend:  # Trend reversal
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Bollinger_Squeeze_Breakout_Volume_12hTrend"
timeframe = "6h"
leverage = 1.0