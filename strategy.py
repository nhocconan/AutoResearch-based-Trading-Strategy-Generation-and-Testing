#/usr/bin/env python3
"""
4h_Supertrend_EMA_200_Trend
Hypothesis: Combines Supertrend (ATR-based trend) with EMA200 filter for trend direction.
Supertrend captures medium-term trend changes, EMA200 filters for long-term bias.
Works in bull via long signals when Supertrend flips up & price > EMA200.
Works in bear via short signals when Supertrend flips down & price < EMA200.
ATR-based stops reduce whipsaw in sideways markets.
Designed for 4h timeframe targeting 20-50 trades/year.
"""

name = "4h_Supertrend_EMA_200_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR(10) for Supertrend
    def calculate_atr(high, low, close, period=10):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = np.zeros_like(close)
        for i in range(len(close)):
            if i < period:
                atr[i] = np.nan
            else:
                atr[i] = np.nanmean(tr[i-period+1:i+1])
        return atr
    
    # EMA calculation
    def calculate_ema(data, period):
        ema = np.full_like(data, np.nan)
        if len(data) < period:
            return ema
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    # Supertrend calculation
    def calculate_supertrend(high, low, close, atr_period=10, multiplier=3.0):
        atr = calculate_atr(high, low, close, atr_period)
        
        # Basic upper and lower bands
        basic_ub = (high + low) / 2 + multiplier * atr
        basic_lb = (high + low) / 2 - multiplier * atr
        
        # Final upper and lower bands
        final_ub = np.full_like(close, np.nan)
        final_lb = np.full_like(close, np.nan)
        
        for i in range(len(close)):
            if i == 0:
                final_ub[i] = basic_ub[i]
                final_lb[i] = basic_lb[i]
            else:
                if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
                    final_ub[i] = basic_ub[i]
                else:
                    final_ub[i] = final_ub[i-1]
                    
                if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
                    final_lb[i] = basic_lb[i]
                else:
                    final_lb[i] = final_lb[i-1]
        
        # Supertrend
        supertrend = np.full_like(close, np.nan)
        for i in range(len(close)):
            if i == 0:
                supertrend[i] = final_ub[i]
            else:
                if supertrend[i-1] == final_ub[i-1] and close[i] <= final_ub[i]:
                    supertrend[i] = final_ub[i]
                elif supertrend[i-1] == final_ub[i-1] and close[i] > final_ub[i]:
                    supertrend[i] = final_lb[i]
                elif supertrend[i-1] == final_lb[i-1] and close[i] >= final_lb[i]:
                    supertrend[i] = final_lb[i]
                elif supertrend[i-1] == final_lb[i-1] and close[i] < final_lb[i]:
                    supertrend[i] = final_ub[i]
                else:
                    supertrend[i] = supertrend[i-1]
        
        return supertrend, atr
    
    # Calculate indicators
    atr_period = 10
    supertrend_multiplier = 3.0
    ema_period = 200
    
    # Supertrend and ATR
    supertrend, atr = calculate_supertrend(high, low, close, atr_period, supertrend_multiplier)
    
    # EMA200
    ema_200 = calculate_ema(close, ema_period)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period, ema_period, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend[i]) or np.isnan(ema_200[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from Supertrend
        # When price > Supertrend, trend is up (Supertrend acts as support)
        # When price < Supertrend, trend is down (Supertrend acts as resistance)
        trend_up = close[i] > supertrend[i]
        trend_down = close[i] < supertrend[i]
        
        # EMA200 filter for long-term bias
        price_above_ema = close[i] > ema_200[i]
        price_below_ema = close[i] < ema_200[i]
        
        if position == 0:
            # Long: Supertrend up AND price > EMA200
            if trend_up and price_above_ema and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Supertrend down AND price < EMA200
            elif trend_down and price_below_ema and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Supertrend flips down OR price < EMA200
            if trend_down or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Supertrend flips up OR price > EMA200
            if trend_up or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals