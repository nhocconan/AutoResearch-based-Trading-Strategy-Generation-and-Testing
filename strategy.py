#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with weekly trend filter
# - Williams %R(14) on 6h identifies overextended price levels
# - Long when %R < -80 (oversold) and weekly close > weekly EMA20 (bullish bias)
# - Short when %R > -20 (overbought) and weekly close < weekly EMA20 (bearish bias)
# - Volume confirmation: current 6h volume > 1.3x 20-period average
# - Fixed position size 0.25 to control drawdown and minimize fee churn
# - Mean reversion works in both bull/bear markets with trend filter
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)

name = "6h_1w_williamsr_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(20)
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 6h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)  # Avoid division by zero
    
    # Align weekly EMA to 6h timeframe (with extra delay for EMA confirmation)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.3x average 6h volume
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when Williams %R returns to -50 (mean reversion)
            if williams_r[i] >= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when Williams %R returns to -50 (mean reversion)
            if williams_r[i] <= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Mean reversion entry with volume confirmation and weekly trend filter
            if volume_confirmed:
                # Long when oversold and weekly bullish
                if williams_r[i] < -80 and close_1w[-1] > ema_20_1w[-1]:
                    position = 1
                    signals[i] = position_size
                # Short when overbought and weekly bearish
                elif williams_r[i] > -20 and close_1w[-1] < ema_20_1w[-1]:
                    position = -1
                    signals[i] = -position_size
    
    return signals