#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Mean Reversion with 1w Trend Filter and Volume Confirmation
# - Primary: 6h timeframe for lower frequency trading to reduce fee drag
# - HTF: 1w for major trend direction (price > SMA50), 1d for oversold/overbought extremes
# - Long: 6h Williams %R < -80 (oversold) + price > 1w SMA50 (bullish major trend) + 1d volume > 1.5x 20-period MA
# - Short: 6h Williams %R > -20 (overbought) + price < 1w SMA50 (bearish major trend) + 1d volume > 1.5x 20-period MA
# - Exit: Williams %R crosses above -50 (for long) or below -50 (for short) - mean reversion to midpoint
# - Position sizing: 0.25 (discrete level to balance return and drawdown)
# - Target: 50-150 total trades over 4 years (12-37/year) - within 6h sweet spot
# - Williams %R is effective at catching reversals in extended moves
# - 1w SMA50 filter ensures we trade with the major trend, reducing counter-trend losses
# - Volume confirmation increases reliability of reversal signals

name = "6h_1w_1d_williamsr_meanrev_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 60 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1w data
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 6h Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_6h = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_6h = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r_6h = (highest_high_6h - close_6h) / (highest_high_6h - lowest_low_6h + 1e-10) * -100
    
    # Calculate 1w SMA(50) for major trend filter
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(williams_r_6h[i]) or 
            np.isnan(sma_50_1w_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) + price > 1w SMA50 (bullish trend) + volume spike
            if (williams_r_6h[i] < -80 and 
                close_6h[i] > sma_50_1w_aligned[i] and 
                volume_1d[i] > 1.5 * volume_ma_20_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R > -20 (overbought) + price < 1w SMA50 (bearish trend) + volume spike
            elif (williams_r_6h[i] > -20 and 
                  close_6h[i] < sma_50_1w_aligned[i] and 
                  volume_1d[i] > 1.5 * volume_ma_20_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Williams %R crosses above -50 (for long) or below -50 (for short)
            # This represents mean reversion to the midpoint of the range
            if position == 1:  # Long position
                if williams_r_6h[i] > -50:  # Crossed above -50, exiting oversold territory
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if williams_r_6h[i] < -50:  # Crossed below -50, exiting overbought territory
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals