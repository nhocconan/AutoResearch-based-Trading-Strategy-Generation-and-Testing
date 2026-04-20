#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1-day volume spike and 1-day trend filter (EMA21)
# Buy when price breaks above 4h Donchian high with daily EMA21 rising and volume spike
# Sell when price breaks below 4h Donchian low with daily EMA21 falling and volume spike
# Volume spike confirms institutional participation. Daily EMA21 filter avoids counter-trend trades.
# Target: 100-200 total trades over 4 years (25-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE for EMA21 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA21 for trend filter
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
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
    
    # Volume spike: 4h volume > 2.0 x 20-period average
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema21_1d_aligned[i]) or np.isnan(atr_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price levels
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        price = close[i]
        
        # Daily trend: rising EMA21 = bullish, falling EMA21 = bearish
        daily_trend_up = ema21_1d_aligned[i] > ema21_1d_aligned[i-1] if i > 0 else False
        daily_trend_down = ema21_1d_aligned[i] < ema21_1d_aligned[i-1] if i > 0 else False
        
        if position == 0:
            # Long: price breaks above upper band, daily trend up, volume spike
            if price > upper_band and daily_trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower band, daily trend down, volume spike
            elif price < lower_band and daily_trend_down and volume_spike[i]:
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

name = "4h_Donchian20_DailyEMA21Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0