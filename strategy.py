#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extremes (oversold/overbought) with volume confirmation and 4h EMA50 trend filter.
# Enter long when Williams %R < -80 (oversold) with volume > 1.8x average and close > 4h EMA50.
# Enter short when Williams %R > -20 (overbought) with volume > 1.8x average and close < 4h EMA50.
# Exit when Williams %R returns to -50 (mean reversion) or opposite extreme is touched.
# Uses Williams %R for mean reversion in ranging markets, filtered by 4h EMA50 to avoid counter-trend trades.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Uses discrete position sizing (0.25) to control risk. Target: 75-200 total trades over 4 years.

name = "4h_WilliamsR_1dExtreme_Volume1.8x_4hEMA50_v1"
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
    
    # Calculate 1d Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Rolling window for Williams %R (14 periods)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 4h timeframe (with 1-bar delay for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 4h data for EMA50 trend filter (primary timeframe, but calculate once before loop)
    # Note: We're already in 4h timeframe, so we can calculate directly
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 4h EMA50 bias
        bullish_bias = close[i] > ema_50[i]
        bearish_bias = close[i] < ema_50[i]
        
        # Williams %R conditions
        williams_r_val = williams_r_aligned[i]
        oversold = williams_r_val < -80
        overbought = williams_r_val > -20
        mean_reversion_exit = abs(williams_r_val + 50) < 5  # Near -50 (mean reversion)
        opposite_extreme = (position == 1 and williams_r_val > -20) or (position == -1 and williams_r_val < -80)
        
        # Entry conditions
        long_entry = oversold and vol_confirm and bullish_bias
        short_entry = overbought and vol_confirm and bearish_bias
        
        # Exit conditions
        long_exit = mean_reversion_exit or opposite_extreme
        short_exit = mean_reversion_exit or opposite_extreme
        
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