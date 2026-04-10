#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d trend filter and volume confirmation
# - Primary: 6h Williams %R(14) for overbought/oversold signals
# - HTF: 1d EMA(50) for trend direction (only trade pullbacks in trend)
# - HTF: 1d volume MA(20) for volume confirmation on signal
# - Long: Williams %R < -80 (oversold) + price > 1d EMA50 + volume > 1.2x MA
# - Short: Williams %R > -20 (overbought) + price < 1d EMA50 + volume > 1.2x MA
# - Exit: Williams %R crosses back above -50 (for long) or below -50 (for short)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: trend filter ensures we trade with higher timeframe momentum,
#   Williams %R captures mean reversion within the trend, volume avoids false signals
# - Target: 60-120 trades over 4 years (15-30/year) to stay within fee drag limits

name = "6h_1d_williamsr_extreme_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 6h data
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 6h Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_6h) / (highest_high_14 - lowest_low_14) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned to 6h)
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Regime conditions
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close_6h[i] > ema_50_1d_aligned[i]
        price_below_ema = close_6h[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: current 1d volume > 1.2x 20-period MA
        volume_confirm = volume_1d_aligned[i] > 1.2 * volume_ma_20_1d_aligned[i]
        
        # Williams %R extreme conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R oversold + uptrend + volume confirmation
            if (oversold and price_above_ema and volume_confirm):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R overbought + downtrend + volume confirmation
            elif (overbought and price_below_ema and volume_confirm):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Williams %R crosses back above -50 (for long) or below -50 (for short)
            if position == 1:  # Long position
                exit_condition = williams_r[i] > -50  # Crossed above -50
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = williams_r[i] < -50  # Crossed below -50
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals