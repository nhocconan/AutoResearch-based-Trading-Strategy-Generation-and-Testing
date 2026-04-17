#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme with 1d EMA200 trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND price > 1d EMA200 (bullish trend) AND volume > 1.5x 20-period average.
Short when Williams %R > -20 (overbought) AND price < 1d EMA200 (bearish trend) AND volume > 1.5x 20-period average.
Exit when Williams %R crosses back above -50 (for long) or below -50 (for short).
Williams %R identifies reversal points in bear market rallies and bull market pullbacks.
1d EMA200 filters for higher timeframe trend alignment to avoid counter-trend trades.
Volume confirmation reduces false signals. Designed for low trade frequency (12-37/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation (14-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R on 1d timeframe: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    period = 14
    highest_high = pd.Series(high_1d).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low_1d).rolling(window=period, min_periods=period).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1d data for EMA200 trend filter
    close_1d_series = pd.Series(close_1d)
    ema_200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate volume average (20-period) on 1d
    volume_1d = df_1d['volume'].values
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        ema_200 = ema_200_1d_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > 1d EMA200 (bullish trend) AND volume > 1.5x avg
            if wr < -80 and price > ema_200 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 1d EMA200 (bearish trend) AND volume > 1.5x avg
            elif wr > -20 and price < ema_200 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 (recovering from oversold)
            if wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 (declining from overbought)
            if wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA200_Volume_Filter"
timeframe = "6h"
leverage = 1.0