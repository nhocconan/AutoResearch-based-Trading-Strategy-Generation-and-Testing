#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w HMA21 trend filter and volume confirmation
# Uses discrete position sizing (0.30) to minimize fee churn. Combines price channel breakouts
# with higher-timeframe trend filtering for robustness in both bull and bear markets.
# Target: 15-25 trades/year per symbol. Works on BTC and ETH as primary targets.

name = "1d_Donchian20_1wHMA21_VolumeSpike_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for HMA21 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate 1w HMA21
    close_1w = df_1w['close'].values
    half_length = 21 // 2
    sqrt_length = int(np.sqrt(21))
    
    wma_half = pd.Series(close_1w).ewm(span=half_length, adjust=False, min_periods=half_length).mean().values
    wma_full = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    raw_hma = 2.0 * wma_half - wma_full
    hma_21_1w = pd.Series(raw_hma).ewm(span=sqrt_length, adjust=False, min_periods=sqrt_length).mean().values
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # Get 1d data for Donchian(20) channels (based on previous 20 days)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels: based on previous 20 days' OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper and lower bands (20-period)
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (using previous 20 days' levels)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Get 1d data for volume EMA(20) for volume confirmation
    vol_1d = df_1d['volume'].values
    vol_ema_20 = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(hma_21_1w_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1d volume > 2.0 x 20-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_20_aligned[i])
        
        # 1w trend: bullish if close > HMA21, bearish if close < HMA21
        bullish_trend = close[i] > hma_21_1w_aligned[i]
        bearish_trend = close[i] < hma_21_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume confirmation + bullish 1w trend
            if (close[i] > donchian_upper_aligned[i] and volume_confirmed and bullish_trend):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian lower + volume confirmation + bearish 1w trend
            elif (close[i] < donchian_lower_aligned[i] and volume_confirmed and bearish_trend):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price falls below Donchian lower OR 1w trend turns bearish
            if close[i] < donchian_lower_aligned[i] or bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price rises above Donchian upper OR 1w trend turns bullish
            if close[i] > donchian_upper_aligned[i] or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals