#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band squeeze breakout with 1d volume spike confirmation.
- Primary timeframe: 6h for breakout detection.
- HTF: 1d Bollinger Band width percentile (20-period) to identify low volatility squeezes.
- Volume: Current 6h volume > 2.0 * 20-period 1d volume MA to confirm breakout validity.
- Entry: Long when price closes above upper BB AND BB width < 20th percentile AND volume spike.
         Short when price closes below lower BB AND BB width < 20th percentile AND volume spike.
- Exit: Opposite BB breakout or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Rationale: Bollinger Band squeezes precede volatility expansions. Works in both bull/bear markets
             as breakouts can occur in either direction. Volume confirmation filters false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands (20-period, 2 std) on 6h
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_band = basis + (2.0 * dev)
    lower_band = basis - (2.0 * dev)
    bb_width = (upper_band - lower_band) / basis  # Normalized width
    
    # Get 1d data for BB width percentile and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Bollinger Band width (20-period)
    df_1d_close = pd.Series(df_1d['close'].values)
    basis_1d = df_1d_close.rolling(window=20, min_periods=20).mean().values
    dev_1d = df_1d_close.rolling(window=20, min_periods=20).std().values
    bb_width_1d = (basis_1d + 2.0*dev_1d - (basis_1d - 2.0*dev_1d)) / basis_1d
    
    # Calculate 20th percentile of BB width over 50-period lookback
    bb_width_percentile = pd.Series(bb_width_1d).rolling(window=50, min_periods=50).quantile(0.20).values
    
    # Calculate 20-period volume MA on 1d
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 6h volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    # Squeeze condition: BB width < 20th percentile
    squeeze_condition = bb_width < bb_width_percentile_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough 1d bars for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(basis[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(bb_width_percentile_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(squeeze_condition[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        upper_bb = upper_band[i]
        lower_bb = lower_band[i]
        is_squeeze = squeeze_condition[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Check for entry signals during Bollinger Band squeeze with volume spike
            if is_squeeze and vol_spike:
                # Bullish breakout: price closes above upper Bollinger Band
                if curr_close > upper_bb:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price closes below lower Bollinger Band
                elif curr_close < lower_bb:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price closes below lower Bollinger Band OR loss of volume confirmation
            if curr_close < lower_bb or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above upper Bollinger Band OR loss of volume confirmation
            if curr_close > upper_bb or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BollingerSqueeze_1dBBWidthPercentile_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0