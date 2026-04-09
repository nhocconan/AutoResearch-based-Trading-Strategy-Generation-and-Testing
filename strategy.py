#!/usr/bin/env python3
# 6h_adaptive_donchian_volume_regime_v1
# Hypothesis: 6h strategy using Donchian breakouts with volume confirmation and volatility regime filter.
# In ranging markets (2025+), price tends to revert from Donchian channels; in trending markets, breakouts continue.
# Volume confirmation filters false breakouts. Uses ATR-based regime detection to adapt parameters.
# Primary timeframe: 6h, HTF: 1d for regime filter and 1w for longer-term bias.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adaptive_donchian_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ATR-based volatility regime (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility percentile (252-period ~ 1 year) for regime detection
    atr_series = pd.Series(atr_14)
    atr_percentile = atr_series.rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile = np.where(np.isnan(atr_percentile), 50.0, atr_percentile)  # Default to median
    
    # Align volatility regime to 6h timeframe
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # 1w HTF data for longer-term bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA(21) for trend bias
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Donchian channels (20-period) on 6h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr_percentile_aligned[i]) or
            np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Regime-based parameters
        vol_regime = atr_percentile_aligned[i]  # 0-100 percentile
        
        # Adaptive Donchian breakout thresholds based on volatility regime
        if vol_regime > 70:  # High volatility regime
            breakout_multiplier = 0.002  # 0.2% buffer
            volume_threshold = 1.5
        elif vol_regime < 30:  # Low volatility regime
            breakout_multiplier = 0.001  # 0.1% buffer
            volume_threshold = 1.2
        else:  # Medium volatility regime
            breakout_multiplier = 0.0015  # 0.15% buffer
            volume_threshold = 1.3
        
        # Adaptive volume threshold
        volume_confirmed = volume[i] > volume_threshold * volume_ma[i]
        
        # Weekly trend bias
        weekly_bias_up = close[i] > ema_21_1w_aligned[i]
        weekly_bias_down = close[i] < ema_21_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price moves below Donchian mid or volume dries up
            if close[i] < donchian_mid[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above Donchian mid or volume dries up
            if close[i] > donchian_mid[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: price breaks above Donchian high with weekly bias up
                if (close[i] > donchian_high[i] * (1 + breakout_multiplier) and 
                    weekly_bias_up and 
                    high[i] > donchian_high[i]):
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian low with weekly bias down
                elif (close[i] < donchian_low[i] * (1 - breakout_multiplier) and 
                      weekly_bias_down and 
                      low[i] < donchian_low[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals