#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extreme levels with 1h EMA50 trend filter and volume confirmation.
# Enter long when 1d Williams %R < -80 (oversold) and price closes above 1h EMA50 with volume > 1.8x average.
# Enter short when 1d Williams %R > -20 (overbought) and price closes below 1h EMA50 with volume > 1.8x average.
# Exit when Williams %R returns to -50 (mean reversion) or opposite extreme is reached.
# Uses discrete sizing (0.25) to control risk and minimize fee churn. Target: 75-200 total trades over 4 years.
# Williams %R identifies exhaustion points; EMA50 provides trend filter for continuation bias.
# Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend).

name = "4h_WilliamsR_1dExtreme_1hEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Rolling 14-period highest high and lowest low
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Williams %R
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    
    # Align Williams %R to 4h timeframe (with 1-bar delay for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 1h data for EMA50 trend filter (HTF)
    df_1h = get_htf_data(prices, '1h')
    
    if len(df_1h) < 50:
        return np.zeros(n)
    
    # Calculate 1h EMA50
    close_1h = df_1h['close'].values
    ema_50_1h = pd.Series(close_1h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_50_1h)
    
    # Calculate 4h volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1h_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 1h EMA50 bias
        bullish_bias = close[i] > ema_50_1h_aligned[i]
        bearish_bias = close[i] < ema_50_1h_aligned[i]
        
        # Williams %R conditions
        williams_r_val = williams_r_aligned[i]
        oversold = williams_r_val < -80
        overbought = williams_r_val > -20
        mean_reversion_exit = abs(williams_r_val + 50) < 5  # Near -50 (neutral)
        
        # Entry conditions
        long_entry = oversold and vol_confirm and bullish_bias
        short_entry = overbought and vol_confirm and bearish_bias
        
        # Exit conditions: mean reversion or opposite extreme
        long_exit = mean_reversion_exit or williams_r_val > -20  # Return to neutral or overbought
        short_exit = mean_reversion_exit or williams_r_val < -80  # Return to neutral or oversold
        
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