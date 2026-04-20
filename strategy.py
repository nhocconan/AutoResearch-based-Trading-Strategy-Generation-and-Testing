#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1-day volume spike and 1-week trend filter (EMA50)
# In bull markets: buy breakout above 20-period high when weekly EMA50 is rising with daily volume spike
# In bear markets: sell breakdown below 20-period low when weekly EMA50 is falling with daily volume spike
# Daily volume spike confirms institutional participation. Weekly EMA50 filter avoids counter-trend trades.
# Target: 75-200 total trades over 4 years (19-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Load daily data ONCE for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate daily volume moving average (20-day)
    volume_ma_20 = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20)
    
    # Calculate 4h Donchian channels (20-period high/low)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high: highest high of last 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low of last 20 periods
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR for stop sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Volume spike: 4h volume > 2.0 x daily average volume (converted to 4h equivalent)
    volume = prices['volume'].values
    # Daily volume in 4h terms: daily volume / 6 (since 6x4h = 1d)
    volume_equivalent = volume_ma_20_aligned / 6.0
    volume_spike = volume > (volume_equivalent * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema50_1w_aligned[i]) or np.isnan(atr_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price levels
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        price = close[i]
        
        # Weekly trend: rising EMA50 = bullish, falling EMA50 = bearish
        weekly_trend_up = ema50_1w_aligned[i] > ema50_1w_aligned[i-1] if i > 0 else False
        weekly_trend_down = ema50_1w_aligned[i] < ema50_1w_aligned[i-1] if i > 0 else False
        
        if position == 0:
            # Long: price breaks above upper band, weekly trend up, volume spike
            if price > upper_band and weekly_trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower band, weekly trend down, volume spike
            elif price < lower_band and weekly_trend_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss (2.5x ATR below entry) or price breaks below lower band
            if price <= entry_price - 2.5 * atr_4h[i] or price < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss (2.5x ATR above entry) or price breaks above upper band
            if price >= entry_price + 2.5 * atr_4h[i] or price > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_WeeklyEMA50Trend_DailyVolumeSpike"
timeframe = "4h"
leverage = 1.0