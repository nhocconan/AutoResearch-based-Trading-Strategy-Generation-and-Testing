#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + Weekly Volatility Filter
# - Williams %R (14) on 6h identifies overbought/oversold conditions
# - Weekly ATR ratio filters for volatility regime: high volatility = mean reversion, low volatility = trend
# - Long when Williams %R < -80 (oversold) AND weekly ATR ratio > 1.2 (high volatility)
# - Short when Williams %R > -20 (overbought) AND weekly ATR ratio > 1.2 (high volatility)
# - Exit when Williams %R returns to -50 (neutral) OR volatility drops (ATR ratio < 0.8)
# - Designed to capture mean reversion in volatile regimes while avoiding choppy low-volatility periods
# - Weekly volatility filter adapts to changing market conditions (bull/bear/range)

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load weekly data for ATR calculation
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate Weekly ATR(14)
    tr1 = high_weekly[1:] - low_weekly[1:]
    tr2 = np.abs(high_weekly[1:] - close_weekly[:-1])
    tr3 = np.abs(low_weekly[1:] - close_weekly[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    atr_weekly = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate Weekly ATR Ratio (current ATR / 20-period average ATR)
    atr_ma_weekly = pd.Series(atr_weekly).rolling(window=20, min_periods=20).mean().values
    atr_ratio_weekly = atr_weekly / atr_ma_weekly
    
    # Align weekly ATR ratio to 6h timeframe
    atr_ratio_weekly_aligned = align_htf_to_ltf(prices, df_weekly, atr_ratio_weekly)
    
    # Calculate Williams %R (14) on 6h timeframe
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_6h) / (highest_high_14 - lowest_low_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup period
        # Skip if NaN in indicators
        if np.isnan(williams_r[i]) or np.isnan(atr_ratio_weekly_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r[i]
        vol_ratio = atr_ratio_weekly_aligned[i]
        
        if position == 0:
            # Enter only in high volatility regimes (avoid choppy low-vol periods)
            if vol_ratio > 1.2:
                # Long: oversold + high volatility
                if wr < -80:
                    signals[i] = 0.25
                    position = 1
                # Short: overbought + high volatility
                elif wr > -20:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: return to neutral OR volatility drops
            if wr > -50 or vol_ratio < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to neutral OR volatility drops
            if wr < -50 or vol_ratio < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_WeeklyVolatilityFilter"
timeframe = "6h"
leverage = 1.0