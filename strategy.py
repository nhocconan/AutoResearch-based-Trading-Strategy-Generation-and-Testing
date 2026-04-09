#!/usr/bin/env python3
# 1d_donchian_breakout_volume_chop_regime_v1
# Hypothesis: Daily Donchian(20) breakout with volume confirmation (>1.5x 20-day avg volume) and chop regime filter (CHOP(14) < 61.8 for trending markets).
# Uses 1-week HMA(21) as higher timeframe trend filter: only take longs when price > weekly HMA, shorts when price < weekly HMA.
# Exits when price crosses back through daily Donchian midpoint or when chop > 61.8 (range market).
# Discrete position sizing (±0.25) to minimize fee churn. Target: 15-25 trades/year (60-100 total over 4 years) to avoid overtrading.
# Works in bull markets via breakout momentum and in bear markets via short breakdowns with volume confirmation.
# Weekly HMA filter ensures alignment with higher timeframe trend, reducing counter-trend trades.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_volume_chop_regime_v1"
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
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index regime filter (14-period)
    atr_period = 14
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_series = pd.Series(tr)
    atr_series = tr_series.rolling(window=atr_period, min_periods=atr_period).mean()
    highest_high = high_series.rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = low_series.rolling(window=atr_period, min_periods=atr_period).min().values
    atr_sum = tr_series.rolling(window=atr_period, min_periods=atr_period).sum().values
    # Avoid division by zero or log of zero
    denominator = highest_high - lowest_low
    chop = np.where(
        (denominator > 0) & (atr_series.values > 0),
        100 * np.log10(atr_sum / np.log10(atr_period) / denominator),
        100.0  # Default to ranging when invalid
    )
    
    # Multi-timeframe: 1-week HMA(21) trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    n_half = int(21 / 2)
    n_sqrt = int(np.sqrt(21))
    
    wma_half_1w = close_1w_s.rolling(window=n_half, min_periods=n_half).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    ).values
    wma_full_1w = close_1w_s.rolling(window=21, min_periods=21).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    ).values
    raw_hma_1w = 2 * wma_half_1w - wma_full_1w
    hma_1w = pd.Series(raw_hma_1w).rolling(window=n_sqrt, min_periods=n_sqrt).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    ).values
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(hma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Regime filter: chop < 61.8 indicates trending market
        trending_market = chop[i] < 61.8
        # HTF trend filter: price above/below 1-week HMA
        htf_uptrend = close[i] > hma_1w_aligned[i]
        htf_downtrend = close[i] < hma_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian midpoint OR chop > 61.8 (range)
            if close[i] < donchian_mid[i] or chop[i] >= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian midpoint OR chop > 61.8 (range)
            if close[i] > donchian_mid[i] or chop[i] >= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume, regime, and HTF confirmation
            bullish_breakout = (close[i] > donchian_high[i]) and volume_confirmed and trending_market and htf_uptrend
            bearish_breakout = (close[i] < donchian_low[i]) and volume_confirmed and trending_market and htf_downtrend
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals