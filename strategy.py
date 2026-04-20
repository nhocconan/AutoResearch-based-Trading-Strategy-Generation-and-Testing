#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1-week volume spike and weekly trend filter (EMA50)
# In bull markets: buy breakout above 20-period high when weekly EMA50 is rising with volume spike
# In bear markets: sell breakdown below 20-period low when weekly EMA50 is falling with volume spike
# Volume spike confirms institutional participation. Weekly EMA50 filter avoids counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 12h Donchian channels (20-period high/low)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high: highest high of last 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low of last 20 periods
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h ATR for stop sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Volume spike: 12h volume > 2.0 x 20-period average (less frequent for 12h)
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema50_1w_aligned[i]) or np.isnan(atr_12h[i]):
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
            if price <= entry_price - 2.5 * atr_12h[i] or price < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss (2.5x ATR above entry) or price breaks above upper band
            if price >= entry_price + 2.5 * atr_12h[i] or price > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_WeeklyEMA50Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0