#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extremes with 1d EMA34 trend filter and volume spike.
# Enter long when Williams %R < -80 (oversold), 1d EMA34 trending up, and volume > 1.8x 20-bar average.
# Enter short when Williams %R > -20 (overbought), 1d EMA34 trending down, and volume > 1.8x 20-bar average.
# Exit when Williams %R crosses above -50 (for long) or below -50 (for short).
# Uses discrete position sizing (0.25) to minimize fee drag while capturing reversals in both bull and bear markets.
# Williams %R is effective in ranging markets (common in 2025 BTC/ETH) and catches mean reversions.
# Volume spike confirms breakout strength; 1d EMA34 ensures alignment with higher timeframe trend.
# Target: 80-150 total trades over 4 years (20-38/year) to avoid excessive fee churn.

name = "4h_WilliamsR_Extremes_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and EMA34
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 4h: Williams %R and EMA34 from completed 1d bars
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA34 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_34_aligned[i] - ema_34_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        # Williams %R conditions
        wr = williams_r_aligned[i]
        wr_oversold = wr < -80
        wr_overbought = wr > -20
        wr_exit_long = wr > -50  # Exit long when %R crosses above -50
        wr_exit_short = wr < -50  # Exit short when %R crosses below -50
        
        # Handle entries and exits
        if wr_oversold and ema_trend_up and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif wr_overbought and ema_trend_down and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and wr_exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and wr_exit_short:
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