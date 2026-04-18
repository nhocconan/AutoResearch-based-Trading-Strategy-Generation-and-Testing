#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout + Volume Spike + Weekly Trend Filter
# Bollinger Band squeeze (low volatility) precedes explosive breakouts in both bull and bear markets.
# Breakout above upper BB or below lower BB with volume surge indicates strong directional move.
# Weekly trend filter (price vs weekly EMA20) ensures trades align with higher timeframe momentum.
# Works in bull markets (breakouts above upper BB in uptrend) and bear markets (breakdowns below lower BB in downtrend).
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
name = "6h_Bollinger_Squeeze_Breakout_Volume_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 6h
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + (2 * bb_std)
    bb_lower = bb_mid - (2 * bb_std)
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band Squeeze: width below 50th percentile of last 50 periods
    bb_width_s = pd.Series(bb_width)
    bb_width_percentile = bb_width_s.rolling(window=50, min_periods=50).quantile(0.5).values
    squeeze = bb_width < bb_width_percentile
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    # Weekly EMA20 for trend direction
    ema_20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    # Volume spike: current volume > 2.5 * 20-period average volume (~5 days on 6h chart)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(ema_20_weekly_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        bb_upper_val = bb_upper[i]
        bb_lower_val = bb_lower[i]
        ema_weekly_val = ema_20_weekly_aligned[i]
        
        if position == 0:
            # Long: Bollinger squeeze breakout above upper band + volume spike + above weekly EMA
            if squeeze[i] and close_val > bb_upper_val and volume_spike[i] and close_val > ema_weekly_val:
                signals[i] = 0.25
                position = 1
            # Short: Bollinger squeeze breakout below lower band + volume spike + below weekly EMA
            elif squeeze[i] and close_val < bb_lower_val and volume_spike[i] and close_val < ema_weekly_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close below middle band (mean reversion) or opposite band touch
            if close_val < bb_mid[i] or close_val <= bb_lower_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close above middle band (mean reversion) or opposite band touch
            if close_val > bb_mid[i] or close_val >= bb_upper_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals