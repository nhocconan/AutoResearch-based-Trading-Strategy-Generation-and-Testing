#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 12h volume confirmation
# - Uses Bollinger Band width percentile to identify low volatility squeeze regimes
# - Breakout occurs when price closes outside Bollinger Bands during low volatility periods
# - Volume confirmation from 12h timeframe: require above average volume for breakout validity
# - Works in both bull and bear markets by capturing volatility expansion after consolidation
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)

name = "6h_12h_bb_squeeze_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Bollinger Bands (20, 2) on 6h
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = ma_20 + 2 * std_20
    lower_band = ma_20 - 2 * std_20
    bb_width = (upper_band - lower_band) / ma_20
    
    # Calculate Bollinger Band width percentile (50-period lookback)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).apply(
        lambda x: np.percentile(x, 20) if len(x) >= 20 else np.nan, raw=True
    ).values
    
    # Squeeze condition: BB width below 20th percentile of recent values
    squeeze_condition = bb_width < bb_width_percentile
    
    # Calculate 12h volume moving average
    volume_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    # Volume confirmation: current 6h volume > 12h volume MA
    volume_confirmation = volume > volume_12h_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ma_20[i]) or np.isnan(std_20[i]) or 
            np.isnan(squeeze_condition[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for breakout signals
            # Check for Bollinger Band squeeze breakout with volume confirmation
            if squeeze_condition[i] and volume_confirmation[i]:
                # Long breakout: price closes above upper Bollinger Band
                if close[i] > upper_band[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below lower Bollinger Band
                elif close[i] < lower_band[i]:
                    position = -1
                    signals[i] = -0.25
        else:  # Position open - look for mean reversion exit
            # Exit when price returns to middle Bollinger Band (mean reversion)
            if position == 1 and close[i] <= ma_20[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] >= ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals