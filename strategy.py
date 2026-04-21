#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with weekly trend filter and volume confirmation
# Long when Williams %R crosses above -20 in weekly uptrend with volume spike
# Short when Williams %R crosses below -80 in weekly downtrend with volume spike
# Williams %R identifies overbought/oversold conditions, weekly trend ensures directional bias
# Volume spike confirms momentum behind the move
# Target: 15-25 trades/year by requiring confluence of extreme %R, trend alignment, and volume
# Works in bull/bear: Weekly trend filter avoids counter-trend trades, %R extremes capture reversals within trend

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for trend direction
    close_weekly = df_weekly['close'].values
    ema34_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to daily timeframe
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Calculate daily Williams %R (14-period)
    high_daily = prices['high'].values
    low_daily = prices['low'].values
    close_daily = prices['close'].values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_daily).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_daily).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_daily) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # start after weekly EMA warmup
        # Skip if data not ready
        if (np.isnan(ema34_weekly_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = volume > 2.0 * vol_ma[i]
        
        # Weekly trend: price above/below EMA34
        weekly_uptrend = price > ema34_weekly_aligned[i]
        weekly_downtrend = price < ema34_weekly_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Long: Williams %R crosses above -20 (exiting oversold) in weekly uptrend
                if williams_r[i] > -20 and williams_r[i-1] <= -20 and weekly_uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -80 (exiting overbought) in weekly downtrend
                elif williams_r[i] < -80 and williams_r[i-1] >= -80 and weekly_downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Williams %R goes below -80 (overbought) or trend changes
                if williams_r[i] < -80 or not weekly_uptrend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Williams %R goes above -20 (oversold) or trend changes
                if williams_r[i] > -20 or not weekly_downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WilliamsR14_WeeklyEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0