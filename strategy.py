#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume confirmation
# - Williams %R(14) from 6h data: oversold < -80, overbought > -20
# - Long when %R crosses above -80 with volume > 1.5x average AND 12h close > EMA50
# - Short when %R crosses below -20 with volume > 1.5x average AND 12h close < EMA50
# - Exit when %R returns to -50 (mean reversion midpoint) or volume drops below average
# - 12h EMA50 trend filter ensures trades align with intermediate trend
# - Volume confirmation prevents false signals in low momentum environments
# - Mean reversion at -50 provides consistent profit-taking
# - Targets 12-25 trades/year (48-100 total over 4 years) to minimize fee drag
# - Williams %R is effective in both trending and ranging markets when combined with trend filter

name = "6h_12h_williamsr_meanreversion_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute Williams %R(14) from 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denom = highest_high - lowest_low
    williams_r = np.full_like(close, -50.0)  # Default to neutral
    mask = denom != 0
    williams_r[mask] = -100 * (highest_high[mask] - close[mask]) / denom[mask]
    
    # Pre-compute 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new mean reversion entries
            # Long entry: %R crosses above -80 (oversold recovery) with volume spike AND 12h uptrend
            if (williams_r[i-1] <= -80 and williams_r[i] > -80 and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema50_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: %R crosses below -20 (overbought rejection) with volume spike AND 12h downtrend
            elif (williams_r[i-1] >= -20 and williams_r[i] < -20 and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema50_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. %R returns to -50 (mean reversion midpoint)
            # 2. Volume drops below average (loss of momentum)
            if position == 1:  # Long position
                if (williams_r[i] >= -50 or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (williams_r[i] <= -50 or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals