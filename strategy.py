#!/usr/bin/env python3
# 1d_donchian_breakout_volume_chop_regime_v1
# Hypothesis: Daily Donchian channel breakout with volume confirmation (>1.5x 20-period average) and choppiness regime filter (CHOP < 38.2 = trending). Enters long when price breaks above Donchian(20) upper band with volume confirmation and trending regime; short when price breaks below Donchian(20) lower band with volume confirmation and trending regime. Exits on opposite Donchian band touch. Uses discrete position sizing (0.25) to limit fee drag. Designed for low turnover (target: 7-25 trades/year) to work in both bull and bear markets by capturing strong trending moves with institutional volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_volume_chop_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_upper = high_s.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_s.rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (14-period) - measures trend vs range
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) / log10(14))
    # Simplified: CHOP < 38.2 = trending, CHOP > 61.8 = ranging
    tr_s = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.append(close[0], close[:-1]))), np.abs(low - np.append(close[0], close[:-1]))))
    atr14 = tr_s.rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    highest_high = high_s.rolling(window=14, min_periods=14).max().values
    lowest_low = low_s.rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14) / np.log10(highest_high - lowest_low) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((highest_high - lowest_low) > 0, chop, 50.0)
    chop = np.where(np.isnan(chop), 50.0, chop)
    
    # 1w HTF trend filter: 50-period EMA on 1w timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(chop[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Trending regime: CHOP < 38.2
        trending_regime = chop[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: price touches or breaks Donchian lower band
            if close[i] <= donchian_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches or breaks Donchian upper band
            if close[i] >= donchian_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with volume confirmation, trending regime, and 1w trend alignment
            if volume_confirmed and trending_regime:
                # Bullish 1w trend: price above 50-period EMA
                bullish_trend = close[i] > ema_50_1w_aligned[i]
                # Bearish 1w trend: price below 50-period EMA
                bearish_trend = close[i] < ema_50_1w_aligned[i]
                
                # Long: price breaks above Donchian upper band with volume confirmation, trending regime, and bullish 1w trend
                if close[i] > donchian_upper[i] and bullish_trend:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian lower band with volume confirmation, trending regime, and bearish 1w trend
                elif close[i] < donchian_lower[i] and bearish_trend:
                    position = -1
                    signals[i] = -0.25
    
    return signals