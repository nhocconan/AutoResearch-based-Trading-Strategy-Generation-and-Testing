# This strategy is designed for 4h timeframe with a focus on the Stochastic RSI indicator combined with price action and volume confirmation.
# The hypothesis is that Stochastic RSI can identify overbought/oversold conditions with sensitivity to recent price action,
# while volume confirmation and price action filters help avoid false signals in choppy markets.
# The strategy aims for low trade frequency to minimize fee drag, targeting 20-50 trades per year.
# It uses a 14-period RSI, then applies Stochastic to the RSI values over 14 periods, with smoothing of 3.
# Long entries occur when Stochastic RSI crosses above 20 from below, with price above EMA20 and volume above average.
# Short entries occur when Stochastic RSI crosses below 80 from above, with price below EMA20 and volume above average.
# Exits are triggered when Stochastic RSI crosses the opposite level (80 for long, 20 for short) or when price crosses EMA20 in the opposite direction.

#!/usr/bin/env python3
"""
4h_Stochastic_RSI_Volume_Filter
Hypothesis: Stochastic RSI identifies momentum extremes while volume and EMA filters improve signal quality.
Designed for low trade frequency (<25/year) to avoid fee drag in choppy markets.
"""

name = "4h_Stochastic_RSI_Volume_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI calculation
    def rsi(close_prices, period=14):
        delta = np.diff(close_prices)
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        roll_up = np.zeros_like(close_prices)
        roll_down = np.zeros_like(close_prices)
        
        roll_up[period] = np.nansum(up[:period])
        roll_down[period] = np.nansum(down[:period])
        
        for i in range(period+1, len(close_prices)):
            roll_up[i] = roll_up[i-1] - (roll_up[i-1] / period) + up[i-1]
            roll_down[i] = roll_down[i-1] - (roll_down[i-1] / period) + down[i-1]
        
        rs = np.where(roll_down != 0, roll_up / roll_down, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        # Set first period values to NaN
        rsi_vals[:period] = np.nan
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    # Stochastic RSI
    def stoch_rsi(rsi_series, stoch_period=14, k_period=3, d_period=3):
        # Calculate Stochastic of RSI
        rsi_min = np.full_like(rsi_series, np.nan)
        rsi_max = np.full_like(rsi_series, np.nan)
        
        for i in range(stoch_period-1, len(rsi_series)):
            start_idx = i - stoch_period + 1
            rsi_slice = rsi_series[start_idx:i+1]
            if np.all(np.isnan(rsi_slice)):
                rsi_min[i] = np.nan
                rsi_max[i] = np.nan
            else:
                rsi_min[i] = np.nanmin(rsi_slice)
                rsi_max[i] = np.nanmax(rsi_slice)
        
        # Avoid division by zero
        denominator = rsi_max - rsi_min
        stoch_rsi_raw = np.where(denominator != 0, (rsi_series - rsi_min) / denominator * 100, 0)
        
        # Smooth with SMA for %K
        k = np.full_like(stoch_rsi_raw, np.nan)
        for i in range(k_period-1, len(stoch_rsi_raw)):
            start_idx = i - k_period + 1
            k_slice = stoch_rsi_raw[start_idx:i+1]
            if np.all(np.isnan(k_slice)):
                k[i] = np.nan
            else:
                k[i] = np.nanmean(k_slice)
        
        # Smooth %K for %D
        d = np.full_like(k, np.nan)
        for i in range(d_period-1, len(k)):
            start_idx = i - d_period + 1
            k_slice = k[start_idx:i+1]
            if np.all(np.isnan(k_slice)):
                d[i] = np.nan
            else:
                d[i] = np.nanmean(k_slice)
                
        return k, d
    
    stoch_k, stoch_d = stoch_rsi(rsi_vals, 14, 3, 3)
    
    # EMA20 for trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: > 1.2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):  # Start after enough data for indicators
        # Skip if any required value is NaN
        if np.isnan(stoch_k[i]) or np.isnan(ema_20[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # LONG: StochK crosses above 20, price above EMA20, volume confirmation
            if stoch_k[i] > 20 and stoch_k[i-1] <= 20 and close[i] > ema_20[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: StochK crosses below 80, price below EMA20, volume confirmation
            elif stoch_k[i] < 80 and stoch_k[i-1] >= 80 and close[i] < ema_20[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: StochK crosses above 80 or price crosses below EMA20
            if stoch_k[i] >= 80 or close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: StochK crosses below 20 or price crosses above EMA20
            if stoch_k[i] <= 20 or close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals