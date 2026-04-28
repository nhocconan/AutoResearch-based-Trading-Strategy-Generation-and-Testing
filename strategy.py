#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R Extreme + 1d EMA34 Trend + Volume Spike
# Williams %R identifies overbought/oversold conditions. Extreme readings below -90 (oversold) or above -10 (overbought) with 1d EMA34 trend alignment and volume spike provide high-probability reversals.
# Long: %R < -90, price > 1d EMA34, volume > 2.0x 20-bar avg
# Short: %R > -10, price < 1d EMA34, volume > 2.0x 20-bar avg
# Exit: %R crosses back above -50 (for long) or below -50 (for short) or opposite extreme
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
# Williams %R is effective in ranging markets and captures reversals in bear rallies.
# Volume confirmation filters weak signals.
# 1d EMA34 provides robust trend filter that works in both bull and bear markets.

name = "12h_WilliamsR_Extreme_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R(14) on 1d data
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 14)  # Ensure sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA trend filter
        ema_trend_up = close[i] > ema_34_1d_aligned[i]
        ema_trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Williams %R levels
        wr = williams_r_aligned[i]
        wr_oversold = wr < -90  # Extreme oversold
        wr_overbought = wr > -10  # Extreme overbought
        wr_exit_long = wr > -50   # Exit long when %R crosses above -50
        wr_exit_short = wr < -50  # Exit short when %R crosses below -50
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R oversold, 1d EMA34 uptrend, volume confirm
            if wr_oversold and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought, 1d EMA34 downtrend, volume confirm
            elif wr_overbought and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on %R recovery or opposite extreme
            if wr_exit_long or wr_overbought:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on %R recovery or opposite extreme
            if wr_exit_short or wr_oversold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals