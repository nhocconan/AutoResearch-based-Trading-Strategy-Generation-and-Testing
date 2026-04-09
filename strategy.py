#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_regime_v1
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and choppiness regime filter.
# Long: Price breaks above 20-period Donchian high + volume > 1.5x 20-period average + CHOP(14) > 61.8 (ranging market -> mean reversion off channel)
# Short: Price breaks below 20-period Donchian low + volume > 1.5x 20-period average + CHOP(14) > 61.8
# Exit: Price returns to opposite Donchian band (long exits below Donchian low, short exits above Donchian high)
# Uses 12h trend filter: only long when 12h close > 12h EMA20, only short when 12h close < 12h EMA20.
# Choppiness filter ensures we mean-revert in ranging markets and avoid whipsaws in strong trends.
# Target: 20-50 trades/year to minimize fee drag while maintaining edge.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_regime_v1"
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
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for choppiness regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for choppiness calculation
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) - sum of true range over 14 periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Sum of absolute price changes over 14 periods
    price_change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    sum_price_change = pd.Series(price_change).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CHOP = 100 * log10(atr_14 / sum_price_change) / log10(14)
    # Avoid division by zero
    chop_raw = np.where(sum_price_change > 0, atr_14 / sum_price_change, 1.0)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    
    # Align choppiness to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    open_12h = df_12h['open'].values
    
    # 12h EMA20 for trend filter
    close_12h_s = pd.Series(close_12h)
    ema_20_12h = close_12h_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 12h EMA20 to 4h
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or
            np.isnan(volume[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(ema_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Choppiness filter: CHOP > 61.8 indicates ranging market (good for mean reversion)
        chop_ranging = chop_aligned[i] > 61.8
        # 12h trend filter: close > EMA20 for uptrend, < EMA20 for downtrend
        trend_12h_up = close_12h[i] > ema_20_12h_aligned[i]  # Note: using raw 12h close for trend
        trend_12h_down = close_12h[i] < ema_20_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to Donchian low
            if close[i] <= lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to Donchian high
            if close[i] >= highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above Donchian high with volume, chop ranging, and 12h uptrend
            if (close[i] > highest_high[i] and    # Break above Donchian high
                volume_confirmed and              # Volume spike
                chop_ranging and                  # Ranging market (mean reversion favorable)
                trend_12h_up):                    # 12h uptrend
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low with volume, chop ranging, and 12h downtrend
            elif (close[i] < lowest_low[i] and   # Break below Donchian low
                  volume_confirmed and           # Volume spike
                  chop_ranging and               # Ranging market (mean reversion favorable)
                  trend_12h_down):               # 12h downtrend
                position = -1
                signals[i] = -0.25
    
    return signals