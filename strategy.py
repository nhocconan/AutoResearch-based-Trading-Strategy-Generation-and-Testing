#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Mean Reversion with Weekly Trend Filter
# Williams %R identifies overbought/oversold conditions on the 6h chart.
# Mean reversion trades are taken when %R crosses above/below -20/-80.
# Trend filter uses weekly EMA50 to only take long in uptrend and short in downtrend.
# Volume spike confirms momentum at reversal points.
# Designed to work in both bull (mean reversion in uptrend) and bear (mean reversion in downtrend) markets.
# Target: 15-30 trades/year (60-120 over 4 years) to avoid excessive trading.
name = "6h_WilliamsR_MeanReversion_WeeklyTrend_Volume"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate Williams %R on 6h data (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Weekly EMA50 for trend filter
    ema50_weekly = pd.Series(df_weekly['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_6h = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_weekly_6h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold) in weekly uptrend with volume spike
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and ema50_weekly_6h[i] > ema50_weekly_6h[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought) in weekly downtrend with volume spike
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and ema50_weekly_6h[i] < ema50_weekly_6h[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses below -50 (momentum loss) or weekly trend turns down
            if williams_r[i] < -50 or ema50_weekly_6h[i] < ema50_weekly_6h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses above -50 (momentum loss) or weekly trend turns up
            if williams_r[i] > -50 or ema50_weekly_6h[i] > ema50_weekly_6h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals