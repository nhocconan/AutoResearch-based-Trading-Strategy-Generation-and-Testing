#!/usr/bin/env python3
"""
4h_Pivotal_Liquidity_Capture
Hypothesis: Price tends to revert to prior day's high/low liquidity zones during low volatility regimes.
In 4h timeframe: 
- Use 1-day high/low as liquidity magnets (prior session highs/lows act as support/resistance)
- Enter long when price pulls back to prior day's low during low volatility (BB width < 30th percentile)
- Enter short when price pulls back to prior day's high during low volatility
- Require volume confirmation (current volume > 1.5x 20-period average)
- Exit when price reaches opposite liquidity level or volatility expands
- Target: 20-40 trades/year per symbol with size 0.25
Works in both bull/bear markets as it captures mean reversion within institutional liquidity zones.
"""

name = "4h_Pivotal_Liquidity_Capture"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1-day data for liquidity levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's high and low liquidity levels
    prior_day_high = df_1d['high'].shift(1).values  # Prior day's high
    prior_day_low = df_1d['low'].shift(1).values    # Prior day's low
    
    # Align to 4h timeframe (prior day's levels available after 1d bar closes)
    prior_day_high_aligned = align_htf_to_ltf(prices, df_1d, prior_day_high)
    prior_day_low_aligned = align_htf_to_ltf(prices, df_1d, prior_day_low)
    
    # Bollinger Bands for volatility regime (20, 2) on 4h close
    close_series = pd.Series(close)
    sma_20 = close_series.rolling(window=20, min_periods=20).mean()
    std_20 = close_series.rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = upper_bb - lower_bb
    
    # Low volatility regime: BB width below 30th percentile
    bb_width_percentile = bb_width.rolling(window=50, min_periods=50).quantile(0.3)
    low_volatility = bb_width < bb_width_percentile
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_confirmation = volume > (1.5 * vol_ma_20.values)
    
    # Distance to liquidity levels (normalized by ATR for adaptive thresholds)
    # Calculate ATR(14) for adaptive distance measurement
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Normalized distance to liquidity levels
    dist_to_high = np.abs(close - prior_day_high_aligned) / atr_14
    dist_to_low = np.abs(close - prior_day_low_aligned) / atr_14
    
    # Entry thresholds: within 0.5 ATR of liquidity level
    near_prior_high = dist_to_high < 0.5
    near_prior_low = dist_to_low < 0.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(prior_day_high_aligned[i]) or np.isnan(prior_day_low_aligned[i]) or
            np.isnan(low_volatility[i]) or np.isnan(volume_confirmation[i]) or
            np.isnan(near_prior_high[i]) or np.isnan(near_prior_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price near prior day's low + low volatility + volume confirmation
            if near_prior_low[i] and low_volatility[i] and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price near prior day's high + low volatility + volume confirmation
            elif near_prior_high[i] and low_volatility[i] and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches prior day's high OR volatility expands
            if near_prior_high[i] or (not low_volatility[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches prior day's low OR volatility expands
            if near_prior_low[i] or (not low_volatility[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals