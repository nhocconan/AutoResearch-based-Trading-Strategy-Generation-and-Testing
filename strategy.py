#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Williams %R for mean reversion in ranging markets
# Williams %R identifies overbought/oversold conditions on weekly timeframe
# Entry when weekly Williams %R shows extreme readings and price touches daily Bollinger Bands
# Bollinger Band squeeze filter ensures we trade during low volatility periods
# Fixed position size of 0.25 to control drawdown and minimize fee churn
# Target: 20-50 trades/year on 1d timeframe (80-200 total over 4 years)

name = "1d_1w_williams_r_bollinger_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Williams %R (14-period)
    highest_high_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_1w - close_1w) / (highest_high_1w - lowest_low_1w)
    williams_r[highest_high_1w == lowest_low_1w] = -50  # Avoid division by zero
    
    # Calculate daily Bollinger Bands (20-period, 2 std)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    # Bollinger Band width for squeeze filter (low volatility regime)
    bb_width = (upper_band - lower_band) / sma_20
    bb_width_ma_50 = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    
    # Align Williams %R to 1d timeframe (with 1-bar delay for completed weekly bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(sma_20[i]) or np.isnan(std_20[i]) or
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(bb_width[i]) or
            np.isnan(bb_width_ma_50[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume > 1.5x average daily volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Bollinger Band squeeze filter: only trade when volatility is low (regime filter)
        volatility_filter = bb_width[i] < bb_width_ma_50[i]
        
        if not (volume_confirmed and volatility_filter):
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when price returns to middle of Bollinger Bands
            if close[i] >= sma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when price returns to middle of Bollinger Bands
            if close[i] <= sma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Mean reversion entry: Williams %R extremes + Bollinger Band touch
            if williams_r_aligned[i] <= -80 and volume_confirmed:  # Oversold
                if close[i] <= lower_band[i]:  # Price touching or below lower Bollinger Band
                    position = 1
                    signals[i] = position_size
            elif williams_r_aligned[i] >= -20 and volume_confirmed:  # Overbought
                if close[i] >= upper_band[i]:  # Price touching or above upper Bollinger Band
                    position = -1
                    signals[i] = -position_size
    
    return signals