#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Williams %R for mean reversion in trending markets.
# Williams %R measures overbought/oversold levels: values below -80 = oversold, above -20 = overbought.
# Long when Williams %R < -80 (oversold) AND price above 20-period EMA (trend filter).
# Short when Williams %R > -20 (overbought) AND price below 20-period EMA.
# Exit when Williams %R returns to -50 (neutral zone) or trend reverses.
# Williams %R works well in ranging markets with clear reversals.
# EMA filter ensures we trade with the intermediate-term trend.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for Williams %R and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for Williams %R and EMA calculations
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 20-period EMA for trend filter
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for mean reversion entries
            # Long: oversold (Williams %R < -80) AND price above EMA (uptrend)
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_20_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: overbought (Williams %R > -20) AND price below EMA (downtrend)
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_20_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral (-50) or trend breaks
            if (williams_r_aligned[i] >= -50 or 
                close[i] < ema_20_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to neutral (-50) or trend breaks
            if (williams_r_aligned[i] <= -50 or 
                close[i] > ema_20_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_WilliamsR_EMA20_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0