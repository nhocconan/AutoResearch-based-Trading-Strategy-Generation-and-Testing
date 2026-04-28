#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Enter long when price breaks above Camarilla R3 level with 1d EMA34 uptrend and volume > 1.8x 20-bar average.
# Enter short when price breaks below Camarilla S3 level with 1d EMA34 downtrend and volume > 1.8x 20-bar average.
# Exit when price retraces to the Camarilla H3/L3 levels respectively.
# Uses discrete position sizing (0.28) to limit drawdown and reduce fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).
# Camarilla levels provide intraday support/resistance structure; 1d EMA34 ensures higher timeframe alignment;
# volume spike filters weak breakouts. Works in both bull (strong breakouts) and bear (strong breakdowns).

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v2"
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
    
    # Get 1d data for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 12h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Previous day's high, low, close for Camarilla levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align to 12h
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Camarilla levels
    R3 = prev_close_aligned + (prev_high_aligned - prev_low_aligned) * 1.1 / 4
    S3 = prev_close_aligned - (prev_high_aligned - prev_low_aligned) * 1.1 / 4
    H3 = prev_close_aligned + (prev_high_aligned - prev_low_aligned) * 1.1 / 6
    L3 = prev_close_aligned - (prev_high_aligned - prev_low_aligned) * 1.1 / 6
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient history for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or 
            np.isnan(prev_close_aligned[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or
            np.isnan(H3[i]) or np.isnan(L3[i])):
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
            if price > R3[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.28
                position = 1
            # Short entry: price < S3, EMA34 down, volume confirm
            elif price < S3[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.28
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at H3
            if price <= H3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:  # Short - hold or exit at L3
            if price >= L3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals