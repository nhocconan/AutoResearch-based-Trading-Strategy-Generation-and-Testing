#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and ATR-based stoploss
# Long when price breaks above Donchian(20) upper band AND 12h bullish trend (close > EMA50)
# Short when price breaks below Donchian(20) lower band AND 12h bearish trend (close < EMA50)
# Exit when price reverses to Donchian midpoint OR 12h trend changes
# Uses ATR(14) for dynamic position sizing (0.20-0.30 range) and stoploss
# Target: 20-40 trades/year on 4h timeframe to minimize fee drag while capturing strong trends
# Works in bull markets via longs in bullish 12h trend regime and bear markets via shorts in bearish 12h trend regime
# Volume confirmation is implicit via breakout strength (price must close outside channel)

name = "4h_Donchian20_12hTrend_ATRSizing"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 12h data for HTF trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_12h = close_12h > ema_50_12h
    trend_bearish_12h = close_12h < ema_50_12h
    
    # Align 12h trend to 4h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_12h, trend_bullish_12h.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_12h, trend_bearish_12h.astype(float))
    
    # Calculate Donchian(20) channels
    lookback = 20
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    mid_band = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        upper_band[i] = np.max(high[i-lookback+1:i+1])
        lower_band[i] = np.min(low[i-lookback+1:i+1])
        mid_band[i] = (upper_band[i] + lower_band[i]) / 2
    
    # Calculate ATR(14) for position sizing and stoploss
    atr_period = 14
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = np.zeros(n)
    for i in range(atr_period-1, n):
        if i == atr_period-1:
            atr[i] = np.mean(tr[1:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback, atr_period, 60), n):
        # Skip if any value is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate dynamic position size based on ATR (inverse volatility)
        # Scale position size: higher ATR = smaller position, lower ATR = larger position
        atr_ratio = atr[i] / close[i]  # ATR as percentage of price
        base_size = 0.25
        # Inverse volatility scaling: normalize to 0.5-2.0 range
        vol_scalar = np.clip(0.02 / atr_ratio, 0.5, 2.0)  # Target ~2% ATR
        position_size = base_size * vol_scalar
        position_size = np.clip(position_size, 0.20, 0.30)  # Keep in 0.20-0.30 range
        
        if position == 0:
            # Long conditions: price closes above Donchian upper band AND 12h bullish trend
            if (close[i] > upper_band[i] and 
                trend_bullish_aligned[i] > 0.5):  # 12h bullish trend
                signals[i] = position_size
                position = 1
            # Short conditions: price closes below Donchian lower band AND 12h bearish trend
            elif (close[i] < lower_band[i] and 
                  trend_bearish_aligned[i] > 0.5):  # 12h bearish trend
                signals[i] = -position_size
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian mid band OR 12h trend turns bearish
            if (close[i] < mid_band[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above Donchian mid band OR 12h trend turns bullish
            if (close[i] > mid_band[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size
    
    return signals