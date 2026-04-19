#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with weekly trend filter and volume confirmation.
# Long when weekly trend up (price > weekly EMA50) AND Williams %R(14) < -80 (oversold) AND volume > 1.3x daily average volume
# Short when weekly trend down (price < weekly EMA50) AND Williams %R(14) > -20 (overbought) AND volume > 1.3x daily average volume
# Exit when Williams %R crosses back above -50 (for long) or below -50 (for short)
# Williams %R identifies overextended moves, weekly EMA50 filters trend direction, volume confirms strength.
# Target: 20-30 trades/year per symbol.
name = "4h_WilliamsR_WeeklyTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (EMA50)
    df_weekly = get_htf_data(prices, '1w')
    # Calculate EMA50 on weekly close
    weekly_close = df_weekly['close'].values
    ema50_weekly = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Get daily Williams %R (14-period)
    df_daily = get_htf_data(prices, '1d')
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_daily['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_daily['low']).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - df_daily['close'].values) / (highest_high - lowest_low)) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_daily, williams_r)
    
    # Get daily average volume for confirmation
    vol_ma_daily = pd.Series(df_daily['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_weekly_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        wr = williams_r_aligned[i]
        vol_ma = vol_ma_daily_aligned[i]
        vol = volume[i]
        
        # Trend filter: weekly EMA50 direction
        trend_up = price > ema50_weekly_aligned[i]
        trend_down = price < ema50_weekly_aligned[i]
        
        if position == 0:
            # Long entry: weekly trend up + Williams %R oversold + volume confirmation
            if trend_up and wr < -80 and vol > 1.3 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: weekly trend down + Williams %R overbought + volume confirmation
            elif trend_down and wr > -20 and vol > 1.3 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses back above -50
            if wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses back below -50
            if wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals