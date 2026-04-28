#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extremes with 12h EMA50 trend filter and volume confirmation.
# Enter long when 1d Williams %R < -80 (oversold) with volume > 2.0x average and close > 12h EMA50 (bullish bias).
# Enter short when 1d Williams %R > -20 (overbought) with volume > 2.0x average and close < 12h EMA50 (bearish bias).
# Exit when Williams %R returns to -50 (mean reversion) or opposite extreme is reached.
# Uses discrete position sizing (0.25) to control risk and minimize fee churn. Target: 50-120 total trades over 4 years.
# Works in bull markets (oversold bounces in uptrend) and bear markets (overbought rejections in downtrend).
# Uses 1d Williams %R for extreme reversal signals (proven edge in BTC/ETH) and 12h EMA50 for trend filter (reduces whipsaws).

name = "4h_WilliamsR_Extremes_12hEMA50_VolumeConfirm_v2"
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
    
    # Get 1d data for Williams %R calculation (primary signal)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period Williams %R
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close_1d) / (highest_high - lowest_low)) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 12h data for EMA50 trend filter (HTF trend)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Williams %R conditions
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        mean_reversion_exit = abs(williams_r_aligned[i] + 50) < 5  # Near -50
        
        # Trend filter: 12h EMA50 bias
        bullish_bias = close[i] > ema_50_12h_aligned[i]
        bearish_bias = close[i] < ema_50_12h_aligned[i]
        
        # Entry conditions
        long_entry = oversold and vol_confirm and bullish_bias
        short_entry = overbought and vol_confirm and bearish_bias
        
        # Exit conditions: mean reversion or opposite extreme
        long_exit = mean_reversion_exit or williams_r_aligned[i] > -20
        short_exit = mean_reversion_exit or williams_r_aligned[i] < -80
        
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