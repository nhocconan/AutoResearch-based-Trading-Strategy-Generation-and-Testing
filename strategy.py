#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with volume confirmation and weekly trend filter
# In trending markets, price breaks out of 1-day Donchian channels with volume.
# Weekly trend filter ensures we only trade in the direction of the higher timeframe trend.
# Volume confirmation reduces false breakouts. Designed to work in both bull and bear markets
# by adapting to volatility regime via Donchian breakouts which capture volatility expansion.
# Target: 7-25 trades/year per symbol (~30-100 total over 4 years)

name = "1d_Donchian20_Volume_WeeklyTrend"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on weekly close for trend direction
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1-day Donchian channels (20-period)
    # Upper channel: highest high of last 20 days
    # Lower channel: lowest low of last 20 days
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-day average volume
    vol_series = pd.Series(volume)
    vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need Donchian and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_trend = ema_34_1w_aligned[i]
        
        # Volume confirmation
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Donchian levels
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume and weekly uptrend
            if price > upper and volume_confirmed and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band with volume and weekly downtrend
            elif price < lower and volume_confirmed and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower Donchian band
            if price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper Donchian band
            if price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals