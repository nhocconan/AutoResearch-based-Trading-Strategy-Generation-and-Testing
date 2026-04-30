#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation, active only during 08-20 UTC session.
# Camarilla pivot points identify key intraday support/resistance levels. Break of R3/S3 with 4h trend alignment and volume spike
# captures strong momentum moves while minimizing false breakouts. Session filter reduces noise during low-liquidity hours.
# Uses discrete sizing 0.20 to control risk and fee churn. Target: 60-150 total trades over 4 years (15-37/year).
# Works in bull markets (breakouts continue trend) and bear markets (breakdowns continue downtrend) due to trend filter.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session filter
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1h Camarilla pivot points (R3, S3) from prior 1h bar
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    # Prior 1h OHLC for Camarilla calculation (shift to avoid look-ahead)
    prior_close = df_1h['close'].shift(1).values
    prior_high = df_1h['high'].shift(1).values
    prior_low = df_1h['low'].shift(1).values
    
    # Calculate Camarilla levels
    cam_range = prior_high - prior_low
    r3 = prior_close + (cam_range * 1.1 / 4)
    s3 = prior_close - (cam_range * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe (wait for 1h bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1h, s3)
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, 50)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade between 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Camarilla break and 4h trend filter
            if curr_volume_spike:
                # Bullish: Close breaks above R3 + price above 4h EMA50
                if curr_close > curr_r3 and curr_close > curr_ema_50_4h:
                    signals[i] = 0.20
                    position = 1
                # Bearish: Close breaks below S3 + price below 4h EMA50
                elif curr_close < curr_s3 and curr_close < curr_ema_50_4h:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: Close drops below S3 or loses 4h trend
            if curr_close < curr_s3 or curr_close < curr_ema_50_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Close rises above R3 or loses 4h trend
            if curr_close > curr_r3 or curr_close > curr_ema_50_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals