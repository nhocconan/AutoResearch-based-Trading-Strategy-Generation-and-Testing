#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversion with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1w for EMA trend and weekly bias.
- Williams %R(14) identifies overbought/oversold conditions on 6h chart.
- Entry: Long when %R < -80 (oversold) and price > 1w EMA50 (bullish bias).
         Short when %R > -20 (overbought) and price < 1w EMA50 (bearish bias).
         Requires volume spike (>1.5x 20-period MA) for confirmation.
- Exit: When %R reverts to -50 (mean reversion) or opposite extreme reached.
- Works in bull via buying oversold dips in uptrend, in bear via selling overbought rallies in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_r(high, low, close, length=14):
    """Calculate Williams %R indicator"""
    highest_high = pd.Series(high).rolling(window=length, min_periods=length).max()
    lowest_low = pd.Series(low).rolling(window=length, min_periods=length).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    wr = wr.fillna(-50)  # Neutral value when range is zero
    return wr.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate Williams %R on 6h chart
    wr = calculate_williams_r(high, low, close, 14)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need enough 1w bars for EMA50, 20 for volume MA, 14 for WR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(wr[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for mean reversion signals with volume spike and trend filter
            if volume_spike[i]:
                # Bullish setup: oversold (%R < -80) and price above 1w EMA50 (bullish bias)
                if wr[i] < -80 and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish setup: overbought (%R > -20) and price below 1w EMA50 (bearish bias)
                elif wr[i] > -20 and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: %R reverts to -50 (mean reversion) or becomes overbought
            if wr[i] >= -50:  # Mean reversion exit
                signals[i] = 0.0
                position = 0
            elif wr[i] > -20:  # Oversold condition failed, reverse to short
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: %R reverts to -50 (mean reversion) or becomes oversold
            if wr[i] <= -50:  # Mean reversion exit
                signals[i] = 0.0
                position = 0
            elif wr[i] < -80:  # Overbought condition failed, reverse to long
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1wEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0