#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian breakout with weekly trend filter and volume confirmation.
In bull market: buy breakout above 20-period high when weekly trend is up.
In bear market: sell short breakdown below 20-period low when weekly trend is down.
Volume confirms institutional participation. Weekly trend filter avoids counter-trend trades.
Target: 20-40 trades/year per symbol. Works in both bull (breakouts) and bear (breakdowns).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=np.float64)
    
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full_like(high, np.nan, dtype=np.float64)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(high)):
        atr[i] = (tr[i] + (period-1) * atr[i-1]) / period
    
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend
    wk_close = df_1w['close'].values
    ema_34_1w = pd.Series(wk_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get weekly close price for trend comparison
    wk_close_price = df_1w['close'].values
    wk_close_aligned = align_htf_to_ltf(prices, df_1w, wk_close_price)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = np.full_like(high, np.nan, dtype=np.float64)
    donchian_low = np.full_like(high, np.nan, dtype=np.float64)
    
    for i in range(19, len(high)):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Get daily volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 12h ATR for stoploss
    atr_12h = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20) + EMA (34) + volume MA (20)
    start_idx = max(20, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(wk_close_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Current indicators
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        ema_trend = ema_34_1w_aligned[i]
        weekly_close = wk_close_aligned[i]
        
        # Volume filter: volume > 1.3x daily average
        vol_filter = vol_now > 1.3 * vol_ma
        
        # Trend filter: price above/below weekly EMA34
        bullish_trend = weekly_close > ema_trend
        bearish_trend = weekly_close < ema_trend
        
        if position == 0:
            # Long: breakout above upper band with bullish trend and volume
            if price_now > upper_band and bullish_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: breakdown below lower band with bearish trend and volume
            elif price_now < lower_band and bearish_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: breakdown below lower band or trend change
            if price_now < lower_band or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: breakout above upper band or trend change
            if price_now > upper_band or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_DonchianBreakout_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0