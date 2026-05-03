#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width Regime + 1d RSI(2) Extreme Reversion
# Bollinger Band Width (BBW) identifies regime: low BBW = squeeze (impending breakout),
# high BBW = expansion (trending or volatile). In low volatility regimes (BBW < 20th percentile),
# we mean-revert using 1d RSI(2) extremes: long when RSI(2) < 10, short when RSI(2) > 90.
# Volume confirmation (>1.5x 20-period EMA) filters false breakouts.
# Works in both bull and bear markets by adapting to volatility regime.
# Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag.

name = "6h_BBW_Regime_RSI2_Extreme"
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
    
    # Get 1d data for RSI(2) extreme filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d RSI(2) for extreme reversion signals
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], 100.0)  # Handle division by zero
    rsi_2 = 100.0 - (100.0 / (1.0 + rs))
    rsi_2_values = rsi_2.values
    rsi_2_aligned = align_htf_to_ltf(prices, df_1d, rsi_2_values)
    
    # Calculate Bollinger Band Width (BBW) on 6h for regime detection
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean()
    std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std()
    upper_band = sma + (std_dev * bb_std)
    lower_band = sma - (std_dev * bb_std)
    bb_width = ((upper_band - lower_band) / sma) * 100  # Percentage
    bb_width_values = bb_width.values
    
    # Calculate 20th percentile of BBW for regime threshold (using expanding window)
    bb_width_percentile_20 = pd.Series(bb_width_values).expanding(min_periods=bb_period).quantile(0.20).values
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(bb_period, n):  # Start from bb_period to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(rsi_2_aligned[i]) or np.isnan(bb_width_values[i]) or 
            np.isnan(bb_width_percentile_20[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Regime detection: low BBW = squeeze (mean reversion opportunity)
        low_volatility_regime = bb_width_values[i] < bb_width_percentile_20[i]
        
        # RSI(2) extreme signals from 1d
        rsi_extreme_long = rsi_2_aligned[i] < 10
        rsi_extreme_short = rsi_2_aligned[i] > 90
        
        if position == 0:
            # Enter long in low volatility regime with RSI(2) extreme oversold + volume spike
            if low_volatility_regime and rsi_extreme_long and volume_spike:
                signals[i] = 0.25
                position = 1
            # Enter short in low volatility regime with RSI(2) extreme overbought + volume spike
            elif low_volatility_regime and rsi_extreme_short and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI(2) reverts to neutral (above 50) OR volatility expands (exit regime)
            if rsi_2_aligned[i] > 50 or not low_volatility_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI(2) reverts to neutral (below 50) OR volatility expands (exit regime)
            if rsi_2_aligned[i] < 50 or not low_volatility_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals