#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot levels with breakout and mean reversion logic
# - Mean reversion (fade) at weekly S2/R2 levels when price reaches these levels with overbought/oversold signals
# - Breakout (trend-following) at weekly S3/R3 levels when price breaks through with volume confirmation
# - Uses weekly pivot levels calculated from prior week's range (more stable than daily)
# - Adds 1d RSI(14) filter to avoid fading in strong trends and to confirm breakout momentum
# - Designed to work in ranging markets (fades at S2/R2) and trending markets (breakouts at S3/R3)
# - Weekly pivots provide stronger support/resistance, reducing false signals
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_WeeklyPivot_S2R2_S3R3_1dRSI14"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot levels based on prior week's OHLC
    # Standard pivot: P = (H + L + C) / 3
    # S1 = 2*P - H, R1 = 2*P - L
    # S2 = P - (H - L), R2 = P + (H - L)
    # S3 = S1 - (H - L), R3 = R1 + (H - L)
    prev_week_high = df_1w['high'].shift(1)
    prev_week_low = df_1w['low'].shift(1)
    prev_week_close = df_1w['close'].shift(1)
    
    # Calculate pivot point and weekly range
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    weekly_range = prev_week_high - prev_week_low
    
    # Avoid division by zero
    weekly_range = np.where(weekly_range == 0, 0.0001, weekly_range)
    
    S1 = 2 * pivot - prev_week_high
    R1 = 2 * pivot - prev_week_low
    S2 = pivot - weekly_range
    R2 = pivot + weekly_range
    S3 = S1 - weekly_range
    R3 = R1 + weekly_range
    
    # Align weekly pivot levels to 6h timeframe
    S2_6h = align_htf_to_ltf(prices, df_1w, S2.values)
    R2_6h = align_htf_to_ltf(prices, df_1w, R2.values)
    S3_6h = align_htf_to_ltf(prices, df_1w, S3.values)
    R3_6h = align_htf_to_ltf(prices, df_1w, R3.values)
    
    # 1d RSI(14) for momentum/filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate RSI(14) on daily close
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Volume filters
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.2 * vol_ma_20)  # Volume confirmation
    volume_expansion = volume > np.roll(volume, 1)  # Current volume > previous
    volume_expansion[0] = False
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(S2_6h[i]) or np.isnan(R2_6h[i]) or np.isnan(S3_6h[i]) or np.isnan(R3_6h[i]) or
            np.isnan(rsi_1d_aligned[i]) or np.isnan(volume_filter[i]) or np.isnan(volume_expansion[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Mean reversion at S2/R2: fade when price reaches these levels with RSI extremes
            # Long at S2: price touches S2 and RSI shows oversold (< 30)
            if close[i] <= S2_6h[i] * 1.002 and close[i] >= S2_6h[i] * 0.998 and rsi_1d_aligned[i] < 30:
                signals[i] = 0.25
                position = 1
            # Short at R2: price touches R2 and RSI shows overbought (> 70)
            elif close[i] >= R2_6h[i] * 0.998 and close[i] <= R2_6h[i] * 1.002 and rsi_1d_aligned[i] > 70:
                signals[i] = -0.25
                position = -1
            # Breakout at S3/R3: trend-following when price breaks through with volume
            # Long breakout: price breaks above R3 with volume expansion and RSI > 50
            elif close[i] > R3_6h[i] and volume_expansion[i] and rsi_1d_aligned[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S3 with volume expansion and RSI < 50
            elif close[i] < S3_6h[i] and volume_expansion[i] and rsi_1d_aligned[i] < 50:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: take profit at R1 or stop loss if breaks below S2
            if close[i] >= R1_6h[i]:  # Take profit at R1
                signals[i] = 0.0
                position = 0
            elif close[i] < S2_6h[i]:  # Stop loss if breaks below S2
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: take profit at S1 or stop loss if breaks above R2
            if close[i] <= S1_6h[i]:  # Take profit at S1
                signals[i] = 0.0
                position = 0
            elif close[i] > R2_6h[i]:  # Stop loss if breaks above R2
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals