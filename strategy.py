#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volatility regime filter
# In low volatility regimes (1d ATR percentile < 30): breakout in direction of 1d EMA50 trend
# In high volatility regimes (1d ATR percentile > 70): fade Donchian touches (mean reversion)
# Volume confirmation (>1.5x 20-period EMA) filters low-quality signals
# Discrete sizing (0.25) minimizes fee churn. Target: 75-200 trades over 4 years.
# Strategy adapts to volatility regimes to work in both bull and bear markets.

name = "4h_Donchian20_1dATR_Regime_Volume"
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
    
    # Get 1d data for ATR-based volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR (14-period) with proper min_periods
    tr1 = pd.Series(df_1d['high']) - df_1d['low']
    tr2 = (pd.Series(df_1d['high']) - df_1d['close'].shift(1)).abs()
    tr3 = (pd.Series(df_1d['low']) - df_1d['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean()
    
    # Calculate ATR percentile (50-period lookback) for regime classification
    atr_percentile = atr_14.rolling(window=50, min_periods=30).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align 1d ATR percentile to 4h timeframe (completed 1d bar only)
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Calculate 1d EMA50 for trend direction in low volatility regime
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_up = highest_high.values
    donchian_low = lowest_low.values
    donchian_mid = (donchian_up + donchian_low) / 2
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(atr_percentile_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(donchian_up[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            if atr_percentile_aligned[i] < 30:
                # Low volatility regime: breakout in direction of 1d EMA50 trend
                if close[i] > ema_50_aligned[i]:
                    # Uptrend bias: long on break above Donchian upper
                    if close[i] > donchian_up[i] and volume_confirm:
                        signals[i] = 0.25
                        position = 1
                else:
                    # Downtrend bias: short on break below Donchian lower
                    if close[i] < donchian_low[i] and volume_confirm:
                        signals[i] = -0.25
                        position = -1
            elif atr_percentile_aligned[i] > 70:
                # High volatility regime: fade Donchian touches (mean reversion)
                if close[i] <= donchian_low[i] and volume_confirm:
                    # Long at lower band
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= donchian_up[i] and volume_confirm:
                    # Short at upper band
                    signals[i] = -0.25
                    position = -1
            # In medium volatility regime (30-70), stay flat
        elif position == 1:
            # Exit long: price returns to midpoint OR volatility increases (>60) OR volume drops
            if (close[i] >= donchian_mid[i] or 
                atr_percentile_aligned[i] > 60 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint OR volatility increases (>60) OR volume drops
            if (close[i] <= donchian_mid[i] or 
                atr_percentile_aligned[i] > 60 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals