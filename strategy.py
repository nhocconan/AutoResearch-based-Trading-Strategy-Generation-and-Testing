#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w trend filter (EMA34) and volume spike confirmation.
# Enter long when price breaks above Camarilla R3 level with 1w EMA34 uptrend and volume > 2x 24-bar average.
# Enter short when price breaks below Camarilla S3 level with 1w EMA34 downtrend and volume > 2x 24-bar average.
# Exit when price retraces to the Camarilla pivot point (PP).
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).
# Camarilla levels provide precise intraday support/resistance; 1w EMA34 ensures higher timeframe alignment;
# volume spike filters weak breakouts. Works in both bull (strong breakouts at R3/S3) and bear (strong breakdowns).

name = "12h_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeSpike_v1"
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
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 12h
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Calculate Camarilla levels from previous 1d bar
    # We need previous day's high, low, close - use 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align to 12h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Calculate Camarilla levels
    # PP = (prev_high + prev_low + prev_close) / 3
    pp = (prev_high_aligned + prev_low_aligned + prev_close_aligned) / 3
    # R3 = PP + (high - low) * 1.1 / 4
    r3 = pp + (prev_high_aligned - prev_low_aligned) * 1.1 / 4
    # S3 = PP - (high - low) * 1.1 / 4
    s3 = pp - (prev_high_aligned - prev_low_aligned) * 1.1 / 4
    
    # Volume confirmation: >2x 24-bar average volume (24*12h = 12 days)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Ensure sufficient history for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_24[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(pp[i]) or
            np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or np.isnan(prev_close_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation
        vol_spike = volume_spike[i]
        
        # 1w EMA34 trend: slope over 3 periods
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
            # Long entry: price > R3, EMA34 up, volume spike
            if price > r3[i] and ema_trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: price < S3, EMA34 down, volume spike
            elif price < s3[i] and ema_trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at PP
            if price <= pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at PP
            if price >= pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals