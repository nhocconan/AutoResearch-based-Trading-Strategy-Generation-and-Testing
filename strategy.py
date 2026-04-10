#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume spike confirmation
# - Williams %R(14) < -80 indicates oversold, > -20 indicates overbought
# - Long when %R crosses above -80 from below with volume > 1.5x average AND daily close > daily EMA50
# - Short when %R crosses below -20 from above with volume > 1.5x average AND daily close < daily EMA50
# - Exit when %R returns to -50 level (mean reversion target) or volume drops below average
# - Daily trend filter ensures alignment with major trend across market cycles
# - Volume confirmation prevents false signals
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Williams %R is effective in ranging markets; combined with daily trend/volume filters for quality signals

name = "6h_1d_williamsr_meanreversion_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute Williams %R(14) on 6h data
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Calculate rolling max/min for Williams %R
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Handle division by zero
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(williams_r[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get previous completed 1d bar values (shifted by 1 to avoid look-ahead)
        # For 6h timeframe, there are 4 bars per 1d bar
        if i >= 4:  # Need at least 4 6h bars to get previous day's data
            prev_1d_idx = i - 4
            
            if prev_1d_idx >= 0 and not np.isnan(ema50_1d_aligned[prev_1d_idx]):
                # Daily trend filter: use previous day's close vs EMA50
                daily_uptrend = close_6h[prev_1d_idx] > ema50_1d_aligned[prev_1d_idx]
                daily_downtrend = close_6h[prev_1d_idx] < ema50_1d_aligned[prev_1d_idx]
                
                if position == 0:  # Flat - look for new mean reversion entries
                    # Williams %R crossover signals
                    williams_r_prev = williams_r[i-1] if i > 0 else -50
                    williams_r_curr = williams_r[i]
                    
                    # Long signal: %R crosses above -80 from below (oversold recovery)
                    if (williams_r_prev <= -80 and williams_r_curr > -80 and
                        vol_spike.iloc[i] and daily_uptrend):
                        position = 1
                        signals[i] = 0.25
                    # Short signal: %R crosses below -20 from above (overbought rejection)
                    elif (williams_r_prev >= -20 and williams_r_curr < -20 and
                          vol_spike.iloc[i] and daily_downtrend):
                        position = -1
                        signals[i] = -0.25
                else:  # Have position - look for exit
                    # Exit conditions:
                    # 1. Williams %R returns to -50 level (mean reversion target)
                    # 2. Volume drops below average (loss of momentum)
                    if position == 1:  # Long position
                        if (williams_r[i] >= -50 or vol_normal.iloc[i]):
                            position = 0
                            signals[i] = 0.0
                        else:
                            signals[i] = 0.25  # Hold long
                    elif position == -1:  # Short position
                        if (williams_r[i] <= -50 or vol_normal.iloc[i]):
                            position = 0
                            signals[i] = 0.0
                        else:
                            signals[i] = -0.25  # Hold short
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Not enough data yet, hold flat
            signals[i] = 0.0
    
    return signals