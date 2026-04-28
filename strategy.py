#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes with 6h EMA20 trend filter and volume confirmation.
# Enter long when Williams %R(14) crosses above -80 (oversold bounce) with volume > 1.5x average and close > 6h EMA20.
# Enter short when Williams %R(14) crosses below -20 (overbought rejection) with volume > 1.5x average and close < 6h EMA20.
# Exit when Williams %R returns to -50 (mean reversion midpoint) or opposite extreme is reached.
# Uses discrete position sizing (0.25) to control risk and minimize fee churn. Target: 80-160 total trades over 4 years.
# Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend).
# Williams %R captures momentum exhaustion; EMA20 filters counter-trend noise; volume confirms participation.

name = "6h_WilliamsR_Extremes_6hEMA20_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h EMA20 for trend filter
    close_series = pd.Series(close)
    ema_20_6h = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    # Get 1d data for Williams %R calculation (HTF momentum)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Williams %R(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_6h[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 6h EMA20 bias
        bullish_bias = close[i] > ema_20_6h[i]
        bearish_bias = close[i] < ema_20_6h[i]
        
        # Williams %R conditions
        wr = williams_r_aligned[i]
        wr_prev = williams_r_aligned[i-1] if i > 0 else -50
        
        # Long entry: WR crosses above -80 from below (oversold bounce)
        long_cross_up = wr_prev <= -80 and wr > -80
        # Short entry: WR crosses below -20 from above (overbought rejection)
        short_cross_down = wr_prev >= -20 and wr < -20
        
        # Exit conditions: WR returns to -50 or reaches opposite extreme
        long_exit = wr >= -50  # Exit long when WR reaches midpoint
        short_exit = wr <= -50  # Exit short when WR reaches midpoint
        
        # Entry conditions
        long_entry = long_cross_up and vol_confirm and bullish_bias
        short_entry = short_cross_down and vol_confirm and bearish_bias
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals