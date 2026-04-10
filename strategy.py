#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R Reversal with 1w trend filter and volume confirmation
# - Primary: 12h timeframe for lower frequency and reduced fee drag
# - HTF: 1w for trend direction (avoid counter-trend trades)
# - Long: Williams %R < -80 (oversold) + price > 1w EMA50 + volume > 1.5x 20-period MA
# - Short: Williams %R > -20 (overbought) + price < 1w EMA50 + volume > 1.5x 20-period MA
# - Exit: Williams %R crosses above -50 (long) or below -50 (short)
# - Position sizing: 0.25 (discrete level)
# - Target: 50-120 total trades over 4 years (12-30/year) - within 12h sweet spot
# - Works in bull/bear: Williams %R captures reversals in ranging markets (2025) and pullbacks in trending markets

name = "12h_1w_williamsr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h OHLCV
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 12h Williams %R(14)
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_12h) / (highest_high_14 - lowest_low_14)
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 1w volume moving average (20-period) for volume confirmation
    volume_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_ma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1w volume > 1.5x 20-period MA
        volume_spike = volume_1w[i] > 1.5 * volume_ma_20_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R oversold (< -80) + price above 1w EMA50 + volume spike
            if (williams_r[i] < -80 and close_12h[i] > ema50_1w_aligned[i] and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R overbought (> -20) + price below 1w EMA50 + volume spike
            elif (williams_r[i] > -20 and close_12h[i] < ema50_1w_aligned[i] and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Williams %R crosses above -50 (long) or below -50 (short)
            if position == 1:  # Long position
                exit_condition = williams_r[i] > -50
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = williams_r[i] < -50
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals