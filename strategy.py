#!/usr/bin/env python3
"""
6h_1d_1w_Volume_Regime_Switch
Hypothesis: Switch between trend-following and mean-reversion based on weekly volatility regime.
In low volatility (VIX-like regime): mean-revert at 1d Bollinger Bands with volume confirmation.
In high volatility: trend-follow using 6h Donchian breakout with volume filter.
Weekly ATR percentile determines regime (low vol = ATR < 50th percentile).
Designed to work in both bull and bear by adapting to market conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_Volume_Regime_Switch"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA FOR REGIME ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly ATR(14) for regime detection
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Weekly ATR percentile (50-period lookback) - regime filter
    atr_series = pd.Series(atr_1w)
    atr_percentile = atr_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    atr_regime = align_htf_to_ltf(prices, df_1w, atr_percentile)  # < 0.5 = low vol regime
    
    # === DAILY DATA FOR MEAN REVERSION ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Bollinger Bands (20, 2.0)
    close_series = pd.Series(close_1d)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # Daily volume confirmation - volume > 1.5x average
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_1d / vol_ma
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # === 6H DATA FOR TREND FOLLOWING ===
    # Donchian breakout (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation for breakout
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = volume / vol_ma_6h
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Need enough lookback for indicators
        # Skip if not ready
        if (np.isnan(atr_regime[i]) or np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or np.isnan(vol_ratio_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ratio_6h[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime: low volatility (< 50th percentile) = mean reversion, high volatility = trend follow
        low_vol_regime = atr_regime[i] < 0.5
        
        if low_vol_regime:
            # MEAN REVERSION MODE: BB fade with volume confirmation
            price_at_upper = close[i] >= bb_upper_aligned[i]
            price_at_lower = close[i] <= bb_lower_aligned[i]
            volume_confirm = vol_ratio_aligned[i] > 1.5
            
            # Mean reversion entries
            long_setup = price_at_lower and volume_confirm
            short_setup = price_at_upper and volume_confirm
            
            # Exit when price returns to middle
            bb_middle_aligned = align_htf_to_ltf(prices, df_1d, bb_middle)
            exit_long = close[i] >= bb_middle_aligned[i]
            exit_short = close[i] <= bb_middle_aligned[i]
            
        else:
            # TREND FOLLOWING MODE: Donchian breakout with volume filter
            breakout_up = close[i] > donchian_high[i]
            breakout_down = close[i] < donchian_low[i]
            volume_filter = vol_ratio_6h[i] > 1.3
            
            # Trend following entries
            long_setup = breakout_up and volume_filter
            short_setup = breakout_down and volume_filter
            
            # Exit on opposite breakout or volume failure
            exit_long = breakout_down or (vol_ratio_6h[i] < 1.0)
            exit_short = breakout_up or (vol_ratio_6h[i] < 1.0)
        
        # Execute trades
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals