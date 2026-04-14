#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h ADX trend strength with 4h/1d regime filter and volume confirmation
# Uses ADX(14) > 25 to identify strong trends, filters with 4h/1d EMA direction
# Volume > 1.5x average ensures institutional participation
# Designed to work in both bull and bear markets by only taking strong trend trades
# Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h EMA data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Load 1d EMA data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ADX calculation (14-period)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[1:period])
        # Wilder smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Calculate smoothed +DM, -DM, and TR
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations (ADX needs ~50 periods)
    start = 60
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx[i]) or np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend strength: ADX > 25 indicates strong trend
        strong_trend = adx[i] > 25
        
        # Multi-timeframe alignment: both 4h and 1d EMA agree on direction
        uptrend_4h = price > ema_20_4h_aligned[i]
        uptrend_1d = price > ema_50_1d_aligned[i]
        downtrend_4h = price < ema_20_4h_aligned[i]
        downtrend_1d = price < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: strong uptrend on both timeframes with volume confirmation
            if strong_trend and uptrend_4h and uptrend_1d and vol > 1.5 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: strong downtrend on both timeframes with volume confirmation
            elif strong_trend and downtrend_4h and downtrend_1d and vol > 1.5 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: trend weakens or reverses on either timeframe
            if not (strong_trend and uptrend_4h and uptrend_1d):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: trend weakens or reverses on either timeframe
            if not (strong_trend and downtrend_4h and downtrend_1d):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_ADX_TrendStrength_4h1dEMA_Volume"
timeframe = "1h"
leverage = 1.0