#!/usr/bin/env python3
# 1d_donchian_1w_volume_chop_v1
# Hypothesis: Daily timeframe Donchian channel breakout with weekly trend filter, volume confirmation, and choppiness regime filter.
# Uses 1d timeframe to minimize trade frequency and fee drag. Donchian(20) breakout captures strong momentum moves.
# Weekly trend filter ensures trades align with higher timeframe direction. Volume confirmation filters weak breakouts.
# Choppiness regime filter (CHOP > 61.8) only allows mean-reversion in ranging markets, but since we use breakouts,
# we actually want CHOP < 38.2 (trending) to avoid false breakouts in chop. Designed for 7-25 trades/year (30-100 over 4 years).
# Works in bull/bear markets: breakouts capture strong moves, weekly filter avoids counter-trend fakes, volume confirms conviction.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_1w_volume_chop_v1"
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
    
    # Get 1d data for Donchian calculation (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper = max(high, lookback=20)
    # Donchian lower = min(low, lookback=20)
    high_roll_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (completed daily candle only)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, high_roll_max)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, low_roll_min)
    
    # Get 1w HTF data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 1w EMA to 1d timeframe (completed weekly candle only)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume spike detection (20-period volume average on 1d)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 1.5)  # 1.5x volume average
    
    # Choppiness filter (use 1d data)
    # CHOP = 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    # We'll use a simplified version: if CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    # For breakout strategy, we want trending markets (CHOP < 38.2) to avoid false breakouts
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # align length
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    max_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * (np.log10(atr14 / (max_high14 - min_low14 + 1e-10)) / np.log10(14))
    # Only trade when market is trending (CHOP < 38.2) to avoid false breakouts in ranging markets
    trending_regime = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(trending_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower (20-day low) or weekly trend turns bearish
            if close[i] < donchian_low_aligned[i] or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper (20-day high) or weekly trend turns bullish
            if close[i] > donchian_high_aligned[i] or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above Donchian upper, weekly trend bullish, volume spike, trending regime
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_1w_aligned[i] and 
                vol_spike[i] and 
                trending_regime[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below Donchian lower, weekly trend bearish, volume spike, trending regime
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_1w_aligned[i] and 
                  vol_spike[i] and 
                  trending_regime[i]):
                position = -1
                signals[i] = -0.25
    
    return signals