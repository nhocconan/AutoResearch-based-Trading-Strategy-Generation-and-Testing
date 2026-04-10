#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d ATR-based volatility filter + 1w trend filter
# - Primary signal: Williams %R(14) crosses above -50 (bullish momentum) or below -50 (bearish momentum)
# - Volatility filter: Only trade when 1d ATR(14) > 0.5x 50-period average ATR (avoid low-volatility chop)
# - Trend filter: 1w EMA(50) slope > 0 for longs, < 0 for shorts (trade with weekly trend)
# - Works in bull/bear: In strong trends, momentum continues; in weak trends, volatility filter prevents whipsaws
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines

name = "6h_1d_1w_williamsr_atr_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 60 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d Williams %R(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Pre-compute 1d ATR(14) and its 50-period average for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    avg_atr_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr_14 > (0.5 * avg_atr_50)
    volatility_filter_aligned = align_htf_to_ltf(prices, df_1d, volatility_filter)
    
    # Pre-compute 1w EMA(50) slope for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_slope = ema_50 - np.roll(ema_50, 1)
    ema_slope[0] = 0
    ema_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_slope)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(volatility_filter_aligned[i]) or 
            np.isnan(ema_slope_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (loss of momentum) or trend turns bearish
            if williams_r_aligned[i] < -50 or ema_slope_aligned[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (loss of momentum) or trend turns bullish
            if williams_r_aligned[i] > -50 or ema_slope_aligned[i] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R crosses with volatility and trend confirmation
            if volatility_filter_aligned[i]:
                # Long: Williams %R crosses above -50 with bullish weekly trend
                if williams_r_aligned[i] > -50 and williams_r_aligned[i-1] <= -50 and ema_slope_aligned[i] > 0:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.25
                # Short: Williams %R crosses below -50 with bearish weekly trend
                elif williams_r_aligned[i] < -50 and williams_r_aligned[i-1] >= -50 and ema_slope_aligned[i] < 0:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.25
    
    return signals