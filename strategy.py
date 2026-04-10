#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Mean Reversion with 1d Trend Filter and Volume Spike
# - Primary: 6h timeframe balances trade frequency and fee drag (target: 50-150 trades over 4 years)
# - HTF: 1d for trend direction (EMA34) and volume confirmation
# - Williams %R(14) identifies overbought/oversold conditions on 6h
# - Long: Williams %R < -80 (oversold) + price > 1d EMA34 (uptrend) + volume spike (>1.5x 20-period MA)
# - Short: Williams %R > -20 (overbought) + price < 1d EMA34 (downtrend) + volume spike
# - Exit: Williams %R crosses above -50 (for long) or below -50 (for short) - mean reversion to midpoint
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Mean reversion effective in ranging markets (2025), trend filter captures momentum in trending phases

name = "6h_1d_williamsr_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 6h Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_6h = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_6h = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r_6h = -100 * (highest_high_6h - close_6h) / (highest_high_6h - lowest_low_6h)
    # Handle division by zero (when high == low)
    williams_r_6h = np.where((highest_high_6h - lowest_low_6h) == 0, -50, williams_r_6h)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(williams_r_6h[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_spike = volume_1d[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R oversold (< -80) + price above 1d EMA34 (uptrend) + volume spike
            if (williams_r_6h[i] < -80 and close_6h[i] > ema_34_1d_aligned[i] and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R overbought (> -20) + price below 1d EMA34 (downtrend) + volume spike
            elif (williams_r_6h[i] > -20 and close_6h[i] < ema_34_1d_aligned[i] and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Williams %R crosses above -50 (for long) or below -50 (for short)
            # This represents mean reversion to the midpoint of the range
            if position == 1:  # Long position
                exit_condition = williams_r_6h[i] > -50  # Crossed above -50
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = williams_r_6h[i] < -50  # Crossed below -50
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals