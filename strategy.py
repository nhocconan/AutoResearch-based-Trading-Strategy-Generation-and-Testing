#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above R3 AND close > 4h EMA50 (uptrend) AND volume > 2.0 * 20-bar avg volume
# Short when price breaks below S3 AND close < 4h EMA50 (downtrend) AND volume > 2.0 * 20-bar avg volume
# Exit when price returns to Camarilla pivot point (mean reversion to equilibrium)
# Uses discrete sizing 0.20 to control fee drag and drawdown
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Camarilla levels provide precise intraday support/resistance; 4h EMA50 ensures higher-timeframe trend alignment
# Volume spike confirms institutional participation; pivot point exit works in ranging markets
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods

name = "1h_Camarilla_R3S3_4hEMA50_Volume_v1"
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
    
    # Calculate Camarilla levels from previous bar
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    # Pivot = (high + low + close) / 3
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan
    
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range / 2.0
    s3 = prev_close - 1.1 * camarilla_range / 2.0
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    close_4h_series = pd.Series(close_4h)
    ema50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1h timeframe (wait for completed HTF bar)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate volume confirmation: volume > 2.0 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(camarilla_pivot[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Camarilla breakout signals with trend and volume filters
            # Long: price breaks above R3 AND uptrend AND volume spike
            if close[i] > r3[i] and close[i] > ema50_4h_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 AND downtrend AND volume spike
            elif close[i] < s3[i] and close[i] < ema50_4h_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to pivot point (mean reversion)
            if close[i] <= camarilla_pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to pivot point (mean reversion)
            if close[i] >= camarilla_pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals