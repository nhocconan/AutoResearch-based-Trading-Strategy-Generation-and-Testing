#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Primary: 4h Williams %R(14) for overbought/oversold conditions
# - HTF: 1d EMA(50) for trend direction + 1d volume confirmation (volume > 1.2x 20-period MA)
# - Long: Williams %R < -80 (oversold) + price above 1d EMA50 + volume confirmation
# - Short: Williams %R > -20 (overbought) + price below 1d EMA50 + volume confirmation
# - Exit: Williams %R returns to -50 (mean reversion midpoint)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Mean reversion in ranging markets, trend filter avoids counter-trend in strong trends
# - Target: 75-200 total trades over 4 years (19-50/year) to stay within fee drag limits

name = "4h_1d_williamsr_meanrev_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    # Pre-compute 4h data
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_4h) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA (50-period) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):  # Start after Williams %R warmup period
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.2x 20-period MA
        volume_confirm = volume_1d_aligned[i] > 1.2 * volume_ma_20_1d_aligned[i]
        
        # Williams %R conditions
        williams_oversold = williams_r[i] < -80
        williams_overbought = williams_r[i] > -20
        williams_exit = abs(williams_r[i] + 50) < 5  # Within 5 points of -50 (midpoint)
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Oversold + price above 1d EMA50 + volume confirmation
            if williams_oversold and close_4h[i] > ema_50_1d_aligned[i] and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Overbought + price below 1d EMA50 + volume confirmation
            elif williams_overbought and close_4h[i] < ema_50_1d_aligned[i] and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Williams %R returns to midpoint (-50)
            if williams_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals