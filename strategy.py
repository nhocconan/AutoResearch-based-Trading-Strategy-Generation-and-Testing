#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above Camarilla R1 AND close > 4h EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below Camarilla S1 AND close < 4h EMA50 AND volume > 1.5x 20-period average
# Exit when price re-enters Camarilla H-L range OR close crosses 4h EMA50
# Uses 1h primary timeframe with 4h HTF for trend filter to capture intraday moves with controlled frequency
# Discrete sizing (0.20) to limit fee drag and manage drawdown in both bull and bear markets
# Session filter: 08-20 UTC to reduce noise trades
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag
# Camarilla levels provide intraday support/resistance; EMA50 filters for higher-timeframe trend; volume confirms participation

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop for EMA50 trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels on 4h data (based on previous 4h bar)
    # Camarilla: H-L range from previous period
    prev_4h_high = df_4h['high'].shift(1).values  # Previous 4h high
    prev_4h_low = df_4h['low'].shift(1).values    # Previous 4h low
    prev_4h_close = df_4h['close'].shift(1).values # Previous 4h close
    
    # Avoid look-ahead: shift by 1 to use only completed 4h bars
    range_4h = prev_4h_high - prev_4h_low
    camarilla_h5 = prev_4h_close + range_4h * 1.1/2  # Resistance level 5 (R1 equivalent)
    camarilla_l5 = prev_4h_close - range_4h * 1.1/2  # Support level 5 (S1 equivalent)
    camarilla_h3 = prev_4h_close + range_4h * 1.1/4  # Resistance level 3
    camarilla_l3 = prev_4h_close - range_4h * 1.1/4  # Support level 3
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h5)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l5)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_h5_aligned[i]) or 
            np.isnan(camarilla_l5_aligned[i]) or np.isnan(volume_filter[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla H5 AND close > 4h EMA50 AND volume spike
            if (high[i] > camarilla_h5_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla L5 AND close < 4h EMA50 AND volume spike
            elif (low[i] < camarilla_l5_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price re-enters Camarilla H-L range OR close < 4h EMA50 (trend flip)
            if (low[i] < camarilla_h5_aligned[i] and high[i] > camarilla_l5_aligned[i]) or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price re-enters Camarilla H-L range OR close > 4h EMA50 (trend flip)
            if (low[i] < camarilla_h5_aligned[i] and high[i] > camarilla_l5_aligned[i]) or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals