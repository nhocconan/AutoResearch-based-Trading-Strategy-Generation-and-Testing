#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume spike
# - Williams %R(14) < -80 = oversold (long), > -20 = overbought (short) on 6h
# - Entry only when 6h volume > 1.5x 20-bar average (confirms reversal pressure)
# - 1d EMA(50) trend filter: long only when price > EMA50, short only when price < EMA50
# - Exit when Williams %R returns to -50 (mean reversion midpoint)
# - Discrete position sizing 0.25 to minimize fee churn
# - Targets ~20-30 trades/year (80-120 total over 4 years) to avoid fee drag
# - Williams %R is effective in ranging/volatile markets (2022-2025) and catches reversals
# - Volume spike filters false signals, 1d trend filter avoids counter-trend trades

name = "6h_1d_williamsr_meanreversion_volume_trend_v1"
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
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 6h Williams %R(14)
    lookback = 14
    highest_high = prices['high'].rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = prices['low'].rolling(window=lookback, min_periods=lookback).min().values
    close = prices['close'].values
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute 6h volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: Williams %R < -80 (oversold) with volume spike and 1d uptrend
            if (williams_r[i] < -80 and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: Williams %R > -20 (overbought) with volume spike and 1d downtrend
            elif (williams_r[i] > -20 and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when Williams %R returns to -50 (mean reversion midpoint)
            if position == 1 and williams_r[i] > -50:
                position = 0
                signals[i] = 0.0
            elif position == -1 and williams_r[i] < -50:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals