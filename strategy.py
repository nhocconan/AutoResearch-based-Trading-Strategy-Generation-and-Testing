#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation.
# Enter long when price breaks above Camarilla R3 level with 1w EMA50 uptrend and volume > 2x 24-bar average.
# Enter short when price breaks below Camarilla S3 level with 1w EMA50 downtrend and volume > 2x 24-bar average.
# Exit when price retraces to the Camarilla H4/L4 levels (midpoint).
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).
# Camarilla levels provide precise intraday support/resistance; 1w EMA50 ensures higher timeframe alignment;
# volume confirmation filters weak breakouts. Works in both bull (strong breakouts) and bear (strong breakdowns).

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike_v1"
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 12h
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate Camarilla levels from previous 1d bar
    # Need to resample to 1d to get daily OHLC
    from mtf_data import get_htf_data
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    # Where C = (H+L+O)/3 (typical price)
    
    # We'll calculate these per 1d bar and align to 12h
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    rang = df_1d['high'] - df_1d['low']
    
    R3 = typical_price + rang * 1.1 / 4
    S3 = typical_price - rang * 1.1 / 4
    H4 = typical_price + rang * 1.1 / 2
    L4 = typical_price - rang * 1.1 / 2
    
    # Align Camarilla levels to 12h (use previous day's levels)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3.values)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3.values)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4.values)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4.values)
    
    # Volume confirmation: >2x 24-bar average volume (24*12h = 12 days)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Ensure sufficient history for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_24[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1w EMA50 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_50_aligned[i] - ema_50_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > R3, EMA50 up, volume confirm
            if price > R3_aligned[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price < S3, EMA50 down, volume confirm
            elif price < S3_aligned[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at H4
            if price <= H4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at L4
            if price >= L4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals