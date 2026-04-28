#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R breakout with 1d EMA34 trend filter and volume confirmation.
# Enter long when Williams %R crosses above -20 (from oversold) with 1d EMA34 uptrend and volume > 1.8x 24-bar average.
# Enter short when Williams %R crosses below -80 (from overbought) with 1d EMA34 downtrend and volume confirmation.
# Exit when Williams %R returns to -50 (mean reversion) or opposite extreme is reached.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).
# Williams %R is a momentum oscillator that identifies overbought/oversold conditions.
# The 1d EMA34 provides higher-timeframe trend alignment to avoid counter-trend trades.
# Volume confirmation ensures breakouts have sufficient participation.
# This strategy works in both bull and bear markets by following the 1d trend while timing entries with 6h momentum extremes.

name = "6h_WilliamsR_Breakout_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Get 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    
    if len(df_6h) < 14:  # Need at least one complete 6h bar for Williams %R (14-period)
        return np.zeros(n)
    
    # Calculate 6h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_6h['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_6h['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_6h['close'].values) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 6h (shifted by one bar to avoid look-ahead)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:  # Need sufficient data for EMA calculation
        return np.zeros(n)
    
    # Calculate 1d EMA (34-period)
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: >1.8x 24-bar average volume (4 * 6h = 24h ~ 1d)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 1.8 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 34)  # Ensure sufficient history for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA trend filter: price > EMA34 = uptrend, price < EMA34 = downtrend
        ema_trend_up = close[i] > ema_34_aligned[i]
        ema_trend_down = close[i] < ema_34_aligned[i]
        
        williams_r_val = williams_r_aligned[i]
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R crosses above -20 (from oversold), EMA uptrend, volume confirm
            if williams_r_val > -20 and williams_r_aligned[i-1] <= -20 and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -80 (from overbought), EMA downtrend, volume confirm
            elif williams_r_val < -80 and williams_r_aligned[i-1] >= -80 and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at mean reversion or opposite extreme
            # Exit when Williams %R returns to -50 (mean reversion) or reaches overbought (-20) extreme
            if williams_r_val >= -50 or williams_r_val <= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at mean reversion or opposite extreme
            # Exit when Williams %R returns to -50 (mean reversion) or reaches oversold (-80) extreme
            if williams_r_val <= -50 or williams_r_val >= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals