#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1-week EMA(34) as trend filter, 1-day Donchian(20) breakout, and volume confirmation.
# Long when 1w EMA > price (bullish trend), price breaks above 1d Donchian upper band, volume > 1.5x average.
# Short when 1w EMA < price (bearish trend), price breaks below 1d Donchian lower band, volume > 1.5x average.
# Uses volatility-based position sizing to limit drawdown. Target: 30-100 total trades over 4 years (7-25/year).
# Designed to work in bull (trend follow) and bear (trend still exists in downtrends) by using weekly trend filter.

name = "1d_1wEMA34_1dDonchian_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Get 1d data for Donchian bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1-week EMA(34)
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_bullish = close > ema_1w[-1] if len(ema_1w) > 0 else False  # Will be replaced by aligned version
    
    # 1-day Donchian(20) bands
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1-week EMA to 1d (no extra delay needed for EMA as it's based on closed weekly bar)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    ema_bullish_aligned = ema_1w_aligned < close  # Bullish when price above weekly EMA
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # Volatility-based position sizing (ATR-based)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_factor = np.clip(atr / (close * 0.01), 0.5, 2.0)  # Normalize volatility
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 34  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_bullish_aligned[i]) or np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or np.isnan(vol_ratio[i]) or np.isnan(vol_factor[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly EMA bullish (price > EMA), price breaks above 1d Donchian upper band, volume spike
            if (ema_bullish_aligned[i] and
                close[i] > donchian_high[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25 * vol_factor[i]
                position = 1
                entry_bar = i
            # Short: weekly EMA bearish (price < EMA), price breaks below 1d Donchian lower band, volume spike
            elif (not ema_bullish_aligned[i] and
                  close[i] < donchian_low[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25 * vol_factor[i]
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: trend flip, price breaks below Donchian lower band, or max 20 bars held
            if (not ema_bullish_aligned[i] or 
                close[i] < donchian_low[i] or
                i - entry_bar >= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 * vol_factor[i]
        elif position == -1:
            # Short exit: trend flip, price breaks above Donchian upper band, or max 20 bars held
            if (ema_bullish_aligned[i] or 
                close[i] > donchian_high[i] or
                i - entry_bar >= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25 * vol_factor[i]
    
    return signals