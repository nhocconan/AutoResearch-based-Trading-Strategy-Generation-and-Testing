# 6h Candlestick Body Momentum with 1d Trend Filter
# Hypothesis: Strong directional candles (large body, small wicks) on 6h timeframe
# capture institutional momentum. In bull/bear markets, these candles often signal
# continuation when aligned with 1d trend. Low-frequency trading (~25 trades/year)
# reduces fee drag. Uses body-to-range ratio and EMA trend filter.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h candlestick body strength
    body = np.abs(close - open_price)
    range_hl = high - low
    # Avoid division by zero
    body_ratio = np.where(range_hl > 0, body / range_hl, 0)
    
    # Smooth body ratio to reduce noise
    body_ratio_smooth = pd.Series(body_ratio).ewm(span=3, adjust=False).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    warmup = 40
    
    for i in range(warmup, n):
        # Skip if EMA data not available
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Exit conditions: weak candle or trend change
        if position == 1:  # Long
            if body_ratio_smooth[i] < 0.3 or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short
            if body_ratio_smooth[i] < 0.3 or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions
        if position == 0:
            # Strong bullish candle: close near high, open near low, large body
            is_bullish = close[i] > open_price[i]
            body_near_high = (high[i] - close[i]) < (0.15 * range_hl[i])
            body_near_low = (open_price[i] - low[i]) < (0.15 * range_hl[i]) if is_bullish else False
            strong_bull = is_bullish and body_near_high and body_ratio_smooth[i] > 0.6
            
            # Strong bearish candle: open near high, close near low, large body
            is_bearish = close[i] < open_price[i]
            body_near_low_bear = (close[i] - low[i]) < (0.15 * range_hl[i])
            body_near_high_bear = (open_price[i] - high[i]) > (-0.15 * range_hl[i])
            strong_bear = is_bearish and body_near_low_bear and body_near_high_bear and body_ratio_smooth[i] > 0.6
            
            # Enter long on strong bullish candle in uptrend
            if strong_bull and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Enter short on strong bearish candle in downtrend
            elif strong_bear and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_CandleBodyMomentum_1dEMA34Trend"
timeframe = "6h"
leverage = 1.0