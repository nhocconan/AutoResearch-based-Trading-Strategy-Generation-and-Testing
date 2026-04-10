#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with weekly trend filter
# - Long: Williams %R(14) < -80 (oversold) AND weekly close > weekly EMA(20) (bullish weekly trend)
# - Short: Williams %R(14) > -20 (overbought) AND weekly close < weekly EMA(20) (bearish weekly trend)
# - Exit: Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Designed for 6h timeframe: targets 12-37 trades/year to avoid fee drag
# - Works in bull/bear markets: weekly EMA filter ensures we trade with the higher timeframe trend

name = "6h_1w_williamsr_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_bullish = close_1w > ema_20_1w
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # Pre-compute 6h Williams %R(14)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(weekly_bullish_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (mean reversion complete)
            if williams_r[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (mean reversion complete)
            if williams_r[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries with weekly trend filter
            if weekly_bullish_aligned[i] > 0.5:  # Weekly bullish trend
                # Long: Williams %R < -80 (oversold)
                if williams_r[i] < -80:
                    position = 1
                    signals[i] = 0.25
            else:  # Weekly bearish trend
                # Short: Williams %R > -20 (overbought)
                if williams_r[i] > -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals