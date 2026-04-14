#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ADX trend filter and volume confirmation
# Donchian(20) provides clear breakout levels. ADX(14) > 25 filters for trending markets.
# Volume confirmation ensures institutional participation. Works in bull/bear by
# only taking breakouts in the direction of the 1d ADX trend (ADX > 25 and +DI > -DI for long,
# ADX > 25 and -DI > +DI for short). Target: 20-50 total trades over 4 years (5-12/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14), +DI, -DI on daily timeframe
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        def smooth(values, period):
            smoothed = np.zeros_like(values)
            smoothed[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
            return smoothed
        
        atr = smooth(tr, period)
        plus_di = 100 * smooth(plus_dm, period) / atr
        minus_di = 100 * smooth(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = smooth(dx, period)
        
        return adx, plus_di, minus_di
    
    adx_1d, plus_di_1d, minus_di_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    plus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, plus_di_1d)
    minus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, minus_di_1d)
    
    # Donchian channels (20-period) on 4h
    lookback = 20
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    
    for i in range(lookback, n):
        donchian_high[i] = np.max(high[i-lookback:i])
        donchian_low[i] = np.min(low[i-lookback:i])
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # Donchian lookback and volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(plus_di_1d_aligned[i]) or
            np.isnan(minus_di_1d_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume filter AND ADX trend up
            if (price > donchian_high[i] and 
                adx_1d_aligned[i] > 25 and 
                plus_di_1d_aligned[i] > minus_di_1d_aligned[i] and
                vol > 1.5 * avg_vol[i]):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low with volume filter AND ADX trend down
            elif (price < donchian_low[i] and 
                  adx_1d_aligned[i] > 25 and 
                  minus_di_1d_aligned[i] > plus_di_1d_aligned[i] and
                  vol > 1.5 * avg_vol[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low OR ADX trend weakens
            if price < donchian_low[i] or adx_1d_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high OR ADX trend weakens
            if price > donchian_high[i] or adx_1d_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_ADX_Trend_Volume"
timeframe = "4h"
leverage = 1.0