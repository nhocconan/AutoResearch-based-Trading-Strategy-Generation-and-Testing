#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation
# Camarilla pivot points provide intraday support/resistance levels derived from previous day's range
# Long when price breaks above R1 with 4h EMA50 uptrend and volume > 1.5x 20-bar average
# Short when price breaks below S1 with 4h EMA50 downtrend and volume > 1.5x 20-bar average
# Uses 1h timeframe targeting 15-37 trades/year (~60-150 total over 4 years) to minimize fee drag
# Session filter (08-20 UTC) reduces noise trades outside active market hours
# Signal size: 0.20 (discrete level to minimize fee churn)

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike_v1"
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
    
    # Pre-compute hour filter for session (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla pivot points (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12.0
    s1 = prev_close - camarilla_range * 1.1 / 12.0
    
    # Align 1d Camarilla levels to 1h timeframe (wait for 1d bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # EMA50, volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_50_4h_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R1, price above 4h EMA50, volume spike
            if price > r1_val and price > ema_val and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S1, price below 4h EMA50, volume spike
            elif price < s1_val and price < ema_val and vol_confirm:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stop or reversal
            # Exit on price below S1 (reversal to short side) or below 4h EMA50 (trend change)
            if price < s1_val or price < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit on stop or reversal
            # Exit on price above R1 (reversal to long side) or above 4h EMA50 (trend change)
            if price > r1_val or price > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals