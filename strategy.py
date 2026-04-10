#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Williams %R(14) identifies overbought/oversold conditions
# - In 1d uptrend (price > EMA50), look for mean reversion longs from oversold (%R < -80)
# - In 1d downtrend (price < EMA50), look for mean reversion shorts from overbought (%R > -20)
# - Volume confirmation: current volume > 1.5x 20-bar average to avoid false signals
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)
# - Williams %R works well in ranging markets; 1d trend filter adds directional bias
# - Mean reversion from extremes tends to work in both bull and bear markets

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
    highest_high = prices['high'].rolling(window=14, min_periods=14).max().values
    lowest_low = prices['low'].rolling(window=14, min_periods=14).min().values
    close = prices['close'].values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new mean reversion entries
            # Long when 1d uptrend AND Williams %R oversold (< -80) with volume spike
            if (prices['close'].iloc[i] > ema50_1d_aligned[i] and  # 1d uptrend
                williams_r[i] < -80 and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when 1d downtrend AND Williams %R overbought (> -20) with volume spike
            elif (prices['close'].iloc[i] < ema50_1d_aligned[i] and  # 1d downtrend
                  williams_r[i] > -20 and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when Williams %R returns to neutral range
            # Exit when Williams %R returns to -50 (neutral) or opposite extreme
            exit_signal = False
            if position == 1:  # Long position
                if williams_r[i] >= -50:  # Returned to neutral or overbought
                    exit_signal = True
            elif position == -1:  # Short position
                if williams_r[i] <= -50:  # Returned to neutral or oversold
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals