#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume confirmation, and chop regime filter
# Long when price breaks above Donchian upper band (20-bar high), close > 1d EMA34, volume > 1.8x 20-bar average, and chop < 61.8 (trending regime)
# Short when price breaks below Donchian lower band (20-bar low), close < 1d EMA34, volume > 1.8x 20-bar average, and chop < 61.8 (trending regime)
# Uses Donchian channels for structure, 1d EMA34 for trend filter, volume for momentum confirmation, and chop regime to avoid whipsaws
# Designed for low trade frequency (~19-50/year on 4h) to minimize fee drag
# Works in bull (breakouts with rising volume in trending regime) and bear (breakdowns with rising volume in trending regime)

name = "4h_Donchian20_Volume_1dEMA34_Trend_Chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (1.8x 20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    # Choppiness Index (14-period) to filter for trending regime
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) / log10(14))
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr1 = np.abs(high - np.roll(low, 1))
    tr2 = np.abs(low - np.roll(close, 1))
    tr3 = np.abs(high - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (highest_high - lowest_low)) / np.log10(14)
    chop_filter = chop < 61.8  # Only trade in trending regime (CHOP < 61.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(34, 20, 14) + 1  # EMA34(1d) + Donchian(20) + volume MA(20) + ATR14(14) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_spike[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Donchian upper band, close > 1d EMA34, volume spike, trending regime
            if (close[i] > donchian_high[i] and 
                close[i] > ema_34_aligned[i] and volume_spike[i] and chop_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price < Donchian lower band, close < 1d EMA34, volume spike, trending regime
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_34_aligned[i] and volume_spike[i] and chop_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Donchian lower band or close < 1d EMA34 (trend failure)
            if (close[i] < donchian_low[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > Donchian upper band or close > 1d EMA34 (trend failure)
            if (close[i] > donchian_high[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals