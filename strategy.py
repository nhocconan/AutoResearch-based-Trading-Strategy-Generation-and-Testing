#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d ADX trend filter and volume confirmation
# - Bollinger Band width percentile identifies low volatility squeezes (range contraction)
# - Breakout direction confirmed by 1d ADX > 25 (trending market) and volume spike
# - Long when BB width < 20th percentile AND price breaks above upper band AND 1d ADX > 25 AND volume > 1.5x average
# - Short when BB width < 20th percentile AND price breaks below lower band AND 1d ADX > 25 AND volume > 1.5x average
# - Exit when price returns to middle band (20-period SMA) or opposite band is touched
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~15-25 trades/year (60-100 total over 4 years) to avoid fee drag
# - Bollinger squeeze works well before explosive moves in both bull and bear markets
# - ADX filter ensures we only trade breakouts in trending conditions, reducing false signals
# - Volume confirmation validates breakout strength

name = "6h_1d_bb_squeeze_breakout_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Bollinger Bands (20, 2) on 6h data
    bb_period = 20
    bb_std = 2
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # Middle band = 20-period SMA
    bb_middle = pd.Series(close_6h).rolling(window=bb_period, min_periods=bb_period).mean().values
    # Standard deviation
    bb_std_dev = pd.Series(close_6h).rolling(window=bb_period, min_periods=bb_period).std().values
    # Upper and lower bands
    bb_upper = bb_middle + (bb_std_dev * bb_std)
    bb_lower = bb_middle - (bb_std_dev * bb_std)
    # Bollinger Band Width = (Upper - Lower) / Middle
    bb_width = np.where(bb_middle != 0, (bb_upper - bb_lower) / bb_middle, 0)
    # BB Width percentile rank (20-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, raw=False
    ).values
    
    # Pre-compute 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    # Smoothed values
    tr_period = 14
    atr = pd.Series(tr).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    # Directional Indicators
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bb_width_percentile[i]) or np.isnan(bb_middle[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: BB squeeze breakout up with ADX trend and volume confirmation
            if (bb_width_percentile[i] < 20 and  # BB width in lowest 20th percentile (squeeze)
                close_6h[i] > bb_upper[i] and     # Price breaks above upper band
                adx_aligned[i] > 25 and           # 1d ADX > 25 (trending market)
                vol_spike.iloc[i]):               # Volume > 1.5x average
                position = 1
                signals[i] = 0.25
            # Short signal: BB squeeze breakout down with ADX trend and volume confirmation
            elif (bb_width_percentile[i] < 20 and  # BB width in lowest 20th percentile (squeeze)
                  close_6h[i] < bb_lower[i] and    # Price breaks below lower band
                  adx_aligned[i] > 25 and          # 1d ADX > 25 (trending market)
                  vol_spike.iloc[i]):              # Volume > 1.5x average
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price returns to middle band (mean reversion)
            # 2. Opposite band is touched (reversal signal)
            if position == 1:
                if close_6h[i] <= bb_middle[i] or close_6h[i] >= bb_upper[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:
                if close_6h[i] >= bb_middle[i] or close_6h[i] <= bb_lower[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals