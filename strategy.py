#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining 1d Williams Alligator for trend direction, 4h Donchian(20) breakout for entry, and volume confirmation.
# Williams Alligator (Jaws=13, Teeth=8, Lips=5) confirms trend: price > Teeth = uptrend, price < Teeth = downtrend.
# Long when price > Alligator Teeth, breaks above 4h Donchian upper band, volume > 1.5x average.
# Short when price < Alligator Teeth, breaks below 4h Donchian lower band, volume > 1.5x average.
# Uses ATR-based position sizing (0.25 base) and 20-bar time-based exit to limit drawdown.
# Works in bull (trend follow) and bear (trend still exists in downtrends).

name = "4h_1dAlligator_4hDonchian_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 4h data for Donchian bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Williams Alligator: SMAs of median price (HL/2)
    median_price_1d = (high_1d + low_1d) / 2
    jaws = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values    # Red line
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values     # Green line
    
    # 4h Donchian(20) bands
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Alligator lines to 4h
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Align 4h Donchian bands to 4h
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # ATR(14) for position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    vol_factor = np.clip(atr / (close * 0.01), 0.5, 2.0)  # Normalize volatility
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 34  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(teeth_aligned[i]) or np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(vol_factor[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > Alligator Teeth, price breaks above 4h Donchian upper band, volume spike
            if (close[i] > teeth_aligned[i] and
                close[i] > donchian_high_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25 * vol_factor[i]
                position = 1
                entry_bar = i
            # Short: price < Alligator Teeth, price breaks below 4h Donchian lower band, volume spike
            elif (close[i] < teeth_aligned[i] and
                  close[i] < donchian_low_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25 * vol_factor[i]
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: price < Alligator Teeth, price breaks below Donchian lower band, or max 20 bars held
            if (close[i] < teeth_aligned[i] or 
                close[i] < donchian_low_aligned[i] or
                i - entry_bar >= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 * vol_factor[i]
        elif position == -1:
            # Short exit: price > Alligator Teeth, price breaks above Donchian upper band, or max 20 bars held
            if (close[i] > teeth_aligned[i] or 
                close[i] > donchian_high_aligned[i] or
                i - entry_bar >= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25 * vol_factor[i]
    
    return signals