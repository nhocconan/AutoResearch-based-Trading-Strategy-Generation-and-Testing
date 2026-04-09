#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v3
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and chop regime filter.
# Long when price breaks above Donchian high with volume spike in low chop (trending).
# Short when price breaks below Donchian low with volume spike in low chop.
# Uses 12h EMA for trend filter to avoid counter-trend trades.
# Works in bull/bear: 12h EMA defines trend, Donchian captures breakouts, volume confirms validity.
# Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness index regime filter (14-period)
    # Chop = 100 * log10(sum(atr14) / log10(highest_high - lowest_low)) / log10(14)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - pd.Series(close).shift(1)))
    tr3 = pd.Series(abs(low - pd.Series(close).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14.sum() / np.log10(highest_high - lowest_low)) / np.log10(14) if (highest_high - lowest_low) != 0 else 50
    # Vectorized chop calculation
    atr14_series = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    highest_high_series = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low_series = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr14_series) / np.log10(14) / (np.log10(highest_high_series - lowest_low_series) / np.log10(10) + 1e-10)
    chop = np.where((highest_high_series - lowest_low_series) > 0, 
                    100 * np.log10(atr14_series) / np.log10(highest_high_series - lowest_low_series) / np.log10(14), 50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if np.isnan(ema_12h_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(volume_ma[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend turns bearish
            if close[i] < donchian_low[i] or close[i] < ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend turns bullish
            if close[i] > donchian_high[i] or close[i] > ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            # Chop regime filter: low chop = trending market (chop < 38.2)
            chop_filter = chop[i] < 38.2
            
            if volume_confirmed and chop_filter:
                # Long breakout: price above Donchian high and above 12h EMA (uptrend)
                if close[i] > donchian_high[i] and close[i] > ema_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price below Donchian low and below 12h EMA (downtrend)
                elif close[i] < donchian_low[i] and close[i] < ema_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals