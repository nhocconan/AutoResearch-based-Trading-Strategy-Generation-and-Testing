#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness Index regime filter with 1-day EMA trend and volume confirmation
# Uses Choppiness Index (14) to detect ranging (CHOP > 61.8) vs trending (CHOP < 38.2) markets
# In trending regime: follow 1-day EMA(34) direction with volume confirmation
# In ranging regime: mean revert at Bollinger Bands (20, 2.0)
# Aims to reduce whipsaw in sideways markets while capturing trends
# Target: 20-50 trades/year to minimize fee drag

name = "4h_Choppiness_Regime_EMA_Volume"
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
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    daily_close = df_1d['close'].values
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Bollinger Bands (20, 2.0) on 4h for mean reversion in ranging markets
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    
    # Calculate Choppiness Index (14) on 4h
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # first TR has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low
    
    # Avoid division by zero
    chop = np.full_like(close, 50.0, dtype=float)
    mask = (range_hl != 0) & (~np.isnan(atr_sum)) & (~np.isnan(range_hl))
    chop[mask] = 100 * np.log10(atr_sum[mask] / range_hl[mask]) / np.log10(14)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        chop_val = chop[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter based on regime
            if chop_val < 38.2:  # Trending regime
                # Follow daily EMA trend with volume confirmation
                if close[i] > ema34_1d_val and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < ema34_1d_val and vol_spike:
                    signals[i] = -0.25
                    position = -1
            elif chop_val > 61.8:  # Ranging regime
                # Mean revert at Bollinger Bands
                if close[i] < bb_lower[i] and close[i] > ema34_1d_val:  # Oversold but not strong downtrend
                    signals[i] = 0.25
                    position = 1
                elif close[i] > bb_upper[i] and close[i] < ema34_1d_val:  # Overbought but not strong uptrend
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long
            if chop_val < 38.2:  # Trending: exit if trend reverses
                if close[i] < ema34_1d_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Ranging: exit at mean or opposite band
                if close[i] >= bb_middle[i] or close[i] >= bb_upper[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Exit short
            if chop_val < 38.2:  # Trending: exit if trend reverses
                if close[i] > ema34_1d_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Ranging: exit at mean or opposite band
                if close[i] <= bb_middle[i] or close[i] <= bb_lower[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals