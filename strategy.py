#!/usr/bin/env python3
# mtf_1h_donchian_volume_chop_regime_v1
# Hypothesis: 1h strategy using 4h/1d Donchian breakouts with volume confirmation and Choppiness Index regime filter.
# In bull/bear markets: trade breakouts in trending regimes (CHOP < 38.2), avoid range (CHOP > 61.8).
# Uses 4h for trend direction (price vs 20-period EMA) and 1d for Donchian channels (20-period high/low).
# Volume confirmation (>1.3x 20-period average) filters false breakouts.
# Discrete sizing (0.0, ±0.20) minimizes fee churn. Target: 15-35 trades/year.
# Session filter: 08-20 UTC to avoid low-liquidity hours.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_donchian_volume_chop_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h HTF data for trend direction (EMA20)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d HTF data for Donchian channels (20-period high/low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # 1h Choppiness Index (14-period) for regime filter
    atr_1h = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    hh_1h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_1h = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(hh_1h - ll_1h) * np.sqrt(14)
    chop_num = np.log10(atr_1h.sum()) if hasattr(atr_1h, 'sum') else np.log10(np.nansum(atr_1h))
    # Avoid division by zero and handle NaN
    hh_minus_ll = hh_1h - ll_1h
    chop = np.where(
        (hh_minus_ll > 0) & (~np.isnan(hh_minus_ll)),
        100 * np.log10(atr_1h / hh_minus_ll) / np.log10(14),
        50.0  # neutral when range is zero
    )
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN or not in session
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i]) or np.isnan(chop[i]) or np.isnan(volume_ma[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        trending_regime = chop[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: price falls below 4h EMA20 OR Donchian low breaks
            if close[i] < ema_20_4h_aligned[i] or close[i] < donch_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price rises above 4h EMA20 OR Donchian high breaks
            if close[i] > ema_20_4h_aligned[i] or close[i] > donch_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            if volume_confirmed and trending_regime:
                # Long entry: price above 4h EMA20 AND above 1d Donchian high
                if close[i] > ema_20_4h_aligned[i] and close[i] > donch_high_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short entry: price below 4h EMA20 AND below 1d Donchian low
                elif close[i] < ema_20_4h_aligned[i] and close[i] < donch_low_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals