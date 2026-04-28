#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume spike confirmation.
# Enter long when Alligator jaws (13-period SMMA) cross above teeth (8-period SMMA) and both above lips (5-period SMMA),
# 1d EMA34 trending up, and volume > 1.5x 20-bar average.
# Enter short when jaws cross below teeth and both below lips, 1d EMA34 trending down, and volume > 1.5x 20-bar average.
# Exit when Alligator lines re-cross (jaws cross teeth in opposite direction) or price crosses 1d EMA34.
# Uses discrete position sizing (0.25) to minimize fee drag while maintaining profitability.
# Target: 50-120 total trades over 4 years (12-30/year) to avoid excessive fee churn.
# Williams Alligator identifies trend formation and acceleration; 1d EMA34 filters for higher timeframe trend alignment;
# Volume spike confirms institutional participation in breakouts.

name = "12h_Williams_Alligator_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
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
    
    # Calculate Williams Alligator (SMMA-based) on 12h timeframe
    # SMMA (Smoothed Moving Average) calculation
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CLOSE) / PERIOD
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator lines: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_ma_20[i])):
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
        
        # Williams Alligator conditions
        # Long: Jaw > Teeth > Lips (alligator eating up)
        # Short: Jaw < Teeth < Lips (alligator eating down)
        alligator_long = jaw[i] > teeth[i] and teeth[i] > lips[i]
        alligator_short = jaw[i] < teeth[i] and teeth[i] < lips[i]
        
        # Exit conditions: Alligator lines re-cross or price crosses 1d EMA34
        exit_long = jaw[i] <= teeth[i] or close[i] <= ema_34_aligned[i]
        exit_short = jaw[i] >= teeth[i] or close[i] >= ema_34_aligned[i]
        
        # Handle entries and exits
        if alligator_long and ema_trend_up and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif alligator_short and ema_trend_down and vol_confirm and position >= 0:
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