#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly trend filter and volume confirmation
# Works in bull markets (breakouts) and bear markets (short breakdowns)
# Low frequency: ~10-20 trades/year to avoid fee drag
# Uses discrete position sizing (0.25) to minimize churn
# Volatility filter ensures trades only in sufficient volatility regimes

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily Donchian channels (20-period)
    # Using pandas rolling for efficiency
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ATR(14) for volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First TR
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period median ATR for volatility regime filter
    atr_ma = pd.Series(atr_14).rolling(window=20, min_periods=20).median().values
    
    # Calculate daily average volume for volume filter
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need all indicators
    start_idx = max(20, 20, 14, 20)  # Donchian, ATR, vol avg
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(atr_ma[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_trend_up = close[i] > ema_20_1w_aligned[i]
        weekly_trend_down = close[i] < ema_20_1w_aligned[i]
        
        # Volatility filter: ATR > 20-period median (high volatility regime)
        vol_filter = atr_14[i] > atr_ma[i]
        
        # Volume filter: current volume > 1.5x daily average
        volume_filter = volume[i] > (vol_avg[i] * 1.5)
        
        # Entry conditions
        if position == 0:
            # Long: price breaks above Donchian high + weekly uptrend + volatility + volume
            if (close[i] > donchian_high[i] and weekly_trend_up and 
                vol_filter and volume_filter):
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low + weekly downtrend + volatility + volume
            elif (close[i] < donchian_low[i] and weekly_trend_down and 
                  vol_filter and volume_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend reversal
            if close[i] < donchian_low[i] or not weekly_trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend reversal
            if close[i] > donchian_high[i] or not weekly_trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_WeeklyEMA20_Trend_VolumeFilter"
timeframe = "1d"
leverage = 1.0