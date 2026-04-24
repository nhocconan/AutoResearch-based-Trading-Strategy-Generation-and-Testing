#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla Pivot Breakout with 4h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1h to balance trade frequency and responsiveness.
- HTF: 4h EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Camarilla Pivots: Calculate R3, S3 levels from prior 4h bar (HLC of completed 4h candle).
- Entry: Long when price breaks above R3 with volume > 1.5x 20-bar average AND 4h EMA50 bullish.
         Short when price breaks below S3 with volume > 1.5x 20-bar average AND 4h EMA50 bearish.
- Exit: Reverse signal from opposite direction or time-based exit (max 12 bars hold).
- Signal size: 0.20 discrete to minimize fee churn and control drawdown.
- Session filter: Only trade 08:00-20:00 UTC to avoid low-volume Asian session noise.
- Target: 80-120 total trades over 4 years (20-30/year) for 1h timeframe.
This strategy exploits intraday mean reversion failure points (Camarilla extremes) in the direction of the 4h trend,
with volume confirmation to avoid false breakouts. Works in both bull and bear markets by only taking trend-aligned trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF trend and Camarilla pivots (prior completed bar)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    df_4h_close = df_4h['close'].values
    ema_4h = pd.Series(df_4h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate Camarilla levels from prior 4h bar: R3, S3
    # Typical price = (H + L + C) / 3
    typical_price = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3.0
    typical_price_vals = typical_price.values
    
    # Camarilla R3 = Close + 1.1*(High - Low)
    # Camarilla S3 = Close - 1.1*(High - Low)
    camarilla_r3 = df_4h['close'] + 1.1 * (df_4h['high'] - df_4h['low'])
    camarilla_s3 = df_4h['close'] - 1.1 * (df_4h['high'] - df_4h['low'])
    r3_vals = camarilla_r3.values
    s3_vals = camarilla_s3.values
    
    # Align Camarilla levels to 1h (use prior completed 4h bar)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3_vals)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3_vals)
    
    # Volume spike: current volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Reset bars counter if flat
        if position == 0:
            bars_since_entry = 0
        
        # Check for exit: time-based max hold (12 bars) or reverse signal
        exit_signal = False
        if position != 0:
            bars_since_entry += 1
            if bars_since_entry >= 12:  # Max 12-hour hold
                exit_signal = True
        
        if position == 0 and not exit_signal:
            # Check for entry signals (only in session)
            if in_session[i]:
                # Long: price > R3 AND volume spike AND 4h EMA50 bullish
                if close[i] > r3_aligned[i] and vol_spike[i] and close[i] > ema_4h_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                    bars_since_entry = 0
                # Short: price < S3 AND volume spike AND 4h EMA50 bearish
                elif close[i] < s3_aligned[i] and vol_spike[i] and close[i] < ema_4h_aligned[i]:
                    signals[i] = -0.20
                    position = -1
                    bars_since_entry = 0
        elif position == 1:
            # Exit long: time-based or price < S3 (mean reversion)
            if exit_signal or close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: time-based or price > R3 (mean reversion)
            if exit_signal or close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0