#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1w trend filter and volume confirmation
# - Uses 6h Williams %R(14) for overbought/oversold signals (long < -80, short > -20)
# - Confirms with 1w EMA(21) trend (price above EMA for long, below for short)
# - Adds 6h volume > 1.3x 20-period average to avoid low-volatility false signals
# - Exits when Williams %R returns to -50 (mean reversion target) or opposing signal
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years) to minimize fee drag
# - Works in bull markets (mean reversion in uptrend) and bear markets (mean reversion in downtrend)
# - Williams %R is effective in ranging conditions which dominate bear markets like 2025+

name = "6h_1w_williamsr_meanrev_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w EMA(21) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_21_1w = close_1w_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 1w EMA to 6h
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    denominator = highest_high_14 - lowest_low_14
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    williams_r = -100 * (highest_high_14 - close) / denominator
    
    # 6h Volume > 1.3x 20-period average
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_21_1w_aligned[i]) or
            np.isnan(volume_spike[i]) or close[i] == 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: Williams %R returns to -50 (mean reversion) or short signal
            if williams_r[i] >= -50 or williams_r[i] > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Williams %R returns to -50 (mean reversion) or long signal
            if williams_r[i] <= -50 or williams_r[i] < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extreme with volume confirmation and trend filter
            if (williams_r[i] < -80 and  # Oversold
                volume_spike[i] and      # Volume confirmation
                close[i] > ema_21_1w_aligned[i]):  # Above weekly EMA (uptrend)
                position = 1
                signals[i] = 0.25
            elif (williams_r[i] > -20 and   # Overbought
                  volume_spike[i] and       # Volume confirmation
                  close[i] < ema_21_1w_aligned[i]):  # Below weekly EMA (downtrend)
                position = -1
                signals[i] = -0.25
    
    return signals