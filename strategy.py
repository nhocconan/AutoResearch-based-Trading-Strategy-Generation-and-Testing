#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above upper Donchian channel AND close > 1w EMA34 (uptrend) AND volume > 2.0 * 20-bar avg volume
# Short when price breaks below lower Donchian channel AND close < 1w EMA34 (downtrend) AND volume > 2.0 * 20-bar avg volume
# Exit when price retouches the Donchian midpoint (mean reversion to equilibrium)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# 1w EMA34 provides higher timeframe trend filter to avoid counter-trend trades
# Volume spike threshold of 2.0x reduces false breakouts and lowers trade frequency
# Donchian midpoint exit captures mean reversion after breakout failure in ranging markets
# Works in both bull (trend following) and bear (mean reversion during retracements) markets

name = "1d_Donchian20_1wEMA34_VolumeSpike_v1"
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
    
    # Calculate Donchian channels for 1d timeframe (based on previous 20 bars)
    # Upper channel: highest high of previous 20 bars
    # Lower channel: lowest low of previous 20 bars
    # Middle channel: midpoint of upper and lower
    upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    midpoint = (upper_channel + lower_channel) / 2.0
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 1d timeframe (wait for completed HTF bar)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate volume confirmation: volume > 2.0 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(midpoint[i]) or np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Donchian breakout signals with trend and volume filters
            # Long: Break above upper channel AND uptrend AND volume spike
            if close[i] > upper_channel[i] and close[i] > ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower channel AND downtrend AND volume spike
            elif close[i] < lower_channel[i] and close[i] < ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price retouches Donchian midpoint (mean reversion)
            if close[i] <= midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price retouches Donchian midpoint (mean reversion)
            if close[i] >= midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals