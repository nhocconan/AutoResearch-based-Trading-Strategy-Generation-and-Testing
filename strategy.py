#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and ATR volatility filter
# Long when price breaks above 20-day Donchian high AND 1w bullish trend (close > EMA50) AND ATR(14) > 0.5 * ATR(50) (sufficient volatility)
# Short when price breaks below 20-day Donchian low AND 1w bearish trend (close < EMA50) AND ATR(14) > 0.5 * ATR(50)
# Uses 1w EMA50 for trend filter to reduce whipsaw, targeting 10-25 trades/year on 1d.
# ATR volatility filter ensures we only trade during sufficient market movement, reducing false breakouts in low-volatility periods.
# Works in bull markets via longs in bullish 1w trend regime and bear markets via shorts in bearish 1w trend regime.

name = "1d_Donchian20_1wTrend_ATRVol_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Donchian calculation and ATR - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on 1d data
    # Donchian high = rolling max of high over 20 periods
    # Donchian low = rolling min of low over 20 periods
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) and ATR(50) for volatility filter on 1d data
    # True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan  # First value has no previous close
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close_1d)
    tr3 = np.abs(low_1d - prev_close_1d)
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volatility filter: ATR(14) > 0.5 * ATR(50) ensures sufficient volatility
    vol_filter_1d = atr_14_1d > (0.5 * atr_50_1d)
    
    # Get 1w data for HTF trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_1w = close_1w > ema_50_1w
    trend_bearish_1w = close_1w < ema_50_1w
    
    # Align 1d indicators to 1d timeframe (no alignment needed as we're already on 1d)
    # But we need to align 1w trend to 1d timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, trend_bullish_1w.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, trend_bearish_1w.astype(float))
    
    # Align 1d Donchian levels and volatility filter to 1d timeframe (self-alignment)
    donchian_high_aligned = donchian_high_1d  # Already on 1d timeframe
    donchian_low_aligned = donchian_low_1d    # Already on 1d timeframe
    vol_filter_aligned = vol_filter_1d        # Already on 1d timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(vol_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND 1w bullish trend AND sufficient volatility
            if (close[i] > donchian_high_aligned[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 1w bullish trend
                vol_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND 1w bearish trend AND sufficient volatility
            elif (close[i] < donchian_low_aligned[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 1w bearish trend
                  vol_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian low OR 1w trend turns bearish
            if (close[i] < donchian_low_aligned[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian high OR 1w trend turns bullish
            if (close[i] > donchian_high_aligned[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals