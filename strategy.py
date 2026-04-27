#!/usr/bin/env python3
"""
#100720 - 4h_PriceAction_Pullback_1dTrend_VolumeFilter
Hypothesis: Price action pullback strategy using price rejection at key levels with 1d trend filter and volume confirmation.
Works in bull (pullbacks in uptrend) and bear (pullbacks in downtrend) by trading with the higher timeframe trend.
Target: 20-40 trades/year to minimize fee drag. Uses price action patterns rather than indicator lag.
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
    open_price = prices['open'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1-day EMA20 for trend filter (responsive but smooth)
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate ATR for dynamic thresholds
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Price action signals: rejection candles
        body_size = np.abs(close[i] - open_price[i])
        upper_wick = high[i] - max(close[i], open_price[i])
        lower_wick = min(close[i], open_price[i]) - low[i]
        
        # Bullish rejection: long lower wick, small body (hammer/pin bar)
        bullish_rejection = (lower_wick > body_size * 2) and (body_size > 0)
        # Bearish rejection: long upper wick, small body (shooting star)
        bearish_rejection = (upper_wick > body_size * 2) and (body_size > 0)
        
        # Long condition: bullish rejection + above 1d EMA20 + volume filter
        if (bullish_rejection and 
            close[i] > ema20_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: bearish rejection + below 1d EMA20 + volume filter
        elif (bearish_rejection and 
              close[i] < ema20_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite rejection or trend change
        elif position == 1 and bearish_rejection:
            signals[i] = 0.0
            position = 0
        elif position == -1 and bullish_rejection:
            signals[i] = 0.0
            position = 0
        # Trend filter exit: price crosses 1d EMA20
        elif position == 1 and close[i] < ema20_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema20_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_PriceAction_Pullback_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0