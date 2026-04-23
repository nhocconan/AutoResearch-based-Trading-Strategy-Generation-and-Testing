#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R mean reversion with 12h EMA50 trend filter and volume spike confirmation.
Long when Williams %R crosses above -80 in 12h uptrend with volume > 1.8x 20-period MA.
Short when Williams %R crosses below -20 in 12h downtrend with volume > 1.8x 20-period MA.
Exit when Williams %R returns to -50 or opposite extreme.
Designed for ~20-40 trades/year with strong edge via momentum exhaustion in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R indicator."""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = williams_r.fillna(-50)
    return williams_r.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams %R on 4h data
    williams_r = calculate_williams_r(high, low, close, 14)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # need EMA50, volume MA20, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 12h close > EMA50 = uptrend, close < EMA50 = downtrend
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        trend_up = close_12h_aligned[i] > ema_50_12h_aligned[i]
        trend_down = close_12h_aligned[i] < ema_50_12h_aligned[i]
        
        # Volume filter: 4h volume > 1.8x 20-period MA (strong confirmation)
        vol_filter = volume[i] > 1.8 * vol_ma_20[i]
        
        # Williams %R conditions
        wr = williams_r[i]
        wr_prev = williams_r[i-1] if i > 0 else -50
        
        # Cross above -80 (oversold bounce)
        cross_above_80 = wr > -80 and wr_prev <= -80
        # Cross below -20 (overbought rejection)
        cross_below_20 = wr < -20 and wr_prev >= -20
        # Return to -50 (mean reversion exit)
        return_to_50 = (wr > -45 and wr < -55) if position != 0 else False
        # Opposite extreme exit
        opposite_extreme = (position == 1 and wr < -20) or (position == -1 and wr > -80)
        
        if position == 0:
            # Long: Williams %R crosses above -80 AND uptrend AND volume spike
            if cross_above_80 and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND downtrend AND volume spike
            elif cross_below_20 and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Return to -50 or opposite extreme
            exit_signal = return_to_50 or opposite_extreme
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_MeanReversion_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0