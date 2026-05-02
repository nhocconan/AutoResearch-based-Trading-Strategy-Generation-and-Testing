#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume spike confirmation
# Uses 1h timeframe for signal generation with Camarilla pivot levels from 1h data
# 4h EMA34 provides higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation (1.8x 20-period average) ensures institutional participation
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
# Discrete position sizing (0.20) balances return and risk while minimizing fee churn
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# Works in bull markets via trend-aligned breakouts, in bear via trend filter avoiding false signals
# Camarilla levels provide precise support/resistance for breakout entries

name = "1h_Camarilla_R3_S3_Breakout_4hEMA34_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (avoid low-liquidity Asian session)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1h data ONCE before loop for Camarilla calculation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate Camarilla levels on 1h data (using previous bar's range)
    # Camarilla R3 = close + 1.1*(high-low)*1.1/4
    # Camarilla S3 = close - 1.1*(high-low)*1.1/4
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Previous bar's range for Camarilla calculation
    prev_high = np.roll(high_1h, 1)
    prev_low = np.roll(low_1h, 1)
    prev_close = np.roll(close_1h, 1)
    prev_high[0] = high_1h[0]  # First bar uses current values
    prev_low[0] = low_1h[0]
    prev_close[0] = close_1h[0]
    
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s3)
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Camarilla R3 + price > 4h EMA34 + volume confirm
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema_34_4h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Camarilla S3 + price < 4h EMA34 + volume confirm
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_34_4h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Camarilla S3 or reverse signal
            if close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Camarilla R3 or reverse signal
            if close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals