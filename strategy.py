#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Enter long when price breaks above R3 with 1d EMA34 uptrend and volume > 2x 20-bar average.
# Enter short when price breaks below S3 with 1d EMA34 downtrend and volume > 2x 20-bar average.
# Exit when price retraces to the 1d EMA34 level.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).
# Camarilla levels provide precise intraday support/resistance; 1d EMA34 ensures higher timeframe alignment;
# volume confirmation filters weak breakouts. Works in both bull (strong breakouts) and bear (strong breakdowns).

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 1d Camarilla levels (based on previous 1d bar)
    # R4 = close + 1.5*(high-low)
    # R3 = close + 1.125*(high-low)
    # S3 = close - 1.125*(high-low)
    # S4 = close - 1.5*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    R3 = close_1d_arr + 1.125 * (high_1d - low_1d)
    S3 = close_1d_arr - 1.125 * (high_1d - low_1d)
    
    # Align Camarilla levels to 6h
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    # Exit condition: 1d EMA34
    # ema_34_aligned already calculated
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient history for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i])):
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
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > R3, EMA34 up, volume confirm
            if price > R3_aligned[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price < S3, EMA34 down, volume confirm
            elif price < S3_aligned[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at 1d EMA34
            if price <= ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at 1d EMA34
            if price >= ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals