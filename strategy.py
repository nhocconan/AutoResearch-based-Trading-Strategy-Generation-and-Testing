#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day timeframe with weekly EMA34 trend filter and 1-day price action signals.
# Uses weekly EMA34 to determine primary trend direction (bull/bear) and 1-day price
# action for entries: buy on pullbacks in uptrend, sell on rallies in downtrend.
# Volume confirmation ensures momentum behind moves. Designed for fewer trades
# (target: 10-25/year) to minimize fee drag and work in both bull and bear markets.
# Weekly trend filter prevents counter-trend trading, improving win rate.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for trend filter
    close_weekly = df_weekly['close'].values
    ema_34_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_34_weekly)
    
    # Pre-compute 1-day volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1-day EMA21 for dynamic support/resistance
    ema_21 = prices['close'].ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if np.isnan(ema_34_weekly_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_21[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend filter: price vs weekly EMA34
        uptrend = price > ema_34_weekly_aligned[i]
        downtrend = price < ema_34_weekly_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Long: price pulls back to EMA21 in uptrend
                if uptrend and price <= ema_21[i] * 1.01:  # within 1% above EMA21
                    signals[i] = 0.25
                    position = 1
                # Short: price rallies to EMA21 in downtrend
                elif downtrend and price >= ema_21[i] * 0.99:  # within 1% below EMA21
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions: price moves away from EMA21 or trend changes
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price moves significantly above EMA21 or trend turns down
                if price > ema_21[i] * 1.03 or not uptrend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price moves significantly below EMA21 or trend turns up
                if price < ema_21[i] * 0.97 or not downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_EMA21_Pullback_WeeklyEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0