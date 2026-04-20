#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d ATR-based breakout with weekly trend filter
# - Buy when price breaks above ATR(20) + SMA(50) on 1d timeframe
# - Sell when price breaks below SMA(50) - ATR(20) on 1d timeframe
# - Weekly EMA(50) trend filter: only take long when price > weekly EMA(50), short when price < weekly EMA(50)
# - Uses ATR for volatility-adjusted breakout levels to adapt to changing market conditions
# - Weekly trend filter ensures alignment with higher timeframe trend
# - Designed for 1d timeframe with selective entries to avoid overtrading
# - Target: 7-25 trades per year per symbol (30-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR-based breakout levels on 1d timeframe
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(20) - Average True Range
    atr_20 = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # SMA(50) - Simple Moving Average
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # Calculate upper and lower breakout bands
    upper_band = sma_50 + atr_20
    lower_band = sma_50 - atr_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_20[i]) or np.isnan(sma_50[i]) or \
           np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        price_above_weekly_ema = close[i] > ema_50_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above upper band + price above weekly EMA
            if close[i] > upper_band[i] and price_above_weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower band + price below weekly EMA
            elif close[i] < lower_band[i] and price_below_weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below SMA(50) or weekly trend changes
            if close[i] < sma_50[i] or not price_above_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above SMA(50) or weekly trend changes
            if close[i] > sma_50[i] or not price_below_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_ATRBreakout_WeeklyEMAFilter"
timeframe = "1d"
leverage = 1.0