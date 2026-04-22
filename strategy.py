# 6h_ElderRay_BullBearPower_1dTrend_Filter_v1
# Hypothesis: Elder Ray (Bull/Bear Power) combined with 1-day trend filter
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Long when Bull Power > 0 and Bear Power < 0 (bullish bar) with 1d EMA50 uptrend
# Short when Bear Power > 0 and Bull Power < 0 (bearish bar) with 1d EMA50 downtrend
# Uses 13-period EMA as in classic Elder Ray, suitable for 6h timeframe
# Trend filter avoids counter-trend trades, improving win rate in both bull/bear markets
# Discrete position sizing (0.25) to limit drawdown and reduce trade frequency

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 1-day EMA for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on daily timeframe
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 60-minute data for Elder Ray calculation
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA(13)
    bear_power = ema_13 - low   # EMA(13) - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50_1d = ema_50_1d_aligned[i]
        ema_13_val = ema_13[i]
        bull = bull_power[i]
        bear = bear_power[i]
        
        # Trend filter: 1-day EMA50 slope
        # Uptrend: current EMA50 > previous EMA50
        # Downtrend: current EMA50 < previous EMA50
        if i > 50:
            ema_50_prev = ema_50_1d_aligned[i-1]
            uptrend = ema_50_1d > ema_50_prev
            downtrend = ema_50_1d < ema_50_prev
        else:
            uptrend = True  # Default to allow initial trades
            downtrend = False
        
        # Entry conditions
        if position == 0:
            # Long: Bullish bar (Bull Power > 0 and Bear Power < 0) in uptrend
            if bull > 0 and bear < 0 and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Bearish bar (Bear Power > 0 and Bull Power < 0) in downtrend
            elif bear > 0 and bull < 0 and downtrend:
                signals[i] = -0.25
                position = -1
        
        # Exit conditions
        elif position != 0:
            # Exit long when bearish bar appears OR trend turns down
            if position == 1:
                if bear > 0 or not uptrend:  # Bearish power or trend change
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25  # Hold long
            # Exit short when bullish bar appears OR trend turns up
            elif position == -1:
                if bull > 0 or not downtrend:  # Bullish power or trend change
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0