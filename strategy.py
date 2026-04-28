#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extremes with 1d EMA34 trend filter and volume confirmation.
# Enter long when Williams %R crosses above -80 (oversold bounce), 1d EMA34 trending up, and volume > 1.8x 20-bar average.
# Enter short when Williams %R crosses below -20 (overbought rejection), 1d EMA34 trending down, and volume > 1.8x 20-bar average.
# Exit when Williams %R reaches opposite extreme (-20 for long, -80 for short) or price crosses 1d EMA34.
# Uses discrete position sizing (0.25) to balance return and fee drag.
# Target: 80-160 total trades over 4 years (20-40/year) to avoid excessive fee churn.
# Williams %R identifies momentum extremes; EMA34 filters for 1d trend alignment;
# Volume spike confirms participation in reversals.

name = "6h_WilliamsR_Extremes_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or 
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
        williams_r_prev = williams_r[i-1] if i > 0 else -50
        williams_r_curr = williams_r[i]
        
        # Long: Williams %R crosses above -80 from below
        williams_r_long_signal = (williams_r_prev <= -80) and (williams_r_curr > -80)
        # Short: Williams %R crosses below -20 from above
        williams_r_short_signal = (williams_r_prev >= -20) and (williams_r_curr < -20)
        
        # Exit conditions: Williams %R reaches opposite extreme or price crosses 1d EMA34
        exit_long = williams_r_curr >= -20 or close[i] < ema_34_aligned[i]
        exit_short = williams_r_curr <= -80 or close[i] > ema_34_aligned[i]
        
        # Handle entries and exits
        if williams_r_long_signal and ema_trend_up and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif williams_r_short_signal and ema_trend_down and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
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