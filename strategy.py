#/usr/bin/env python3
# 6h_OrderBlock_Retest_Pattern
# Strategy: Identifies 6h bullish/bearish order blocks using volume imbalance and tests their retest for continuation.
# Bullish OB: candle with close > open and volume > 1.5x avg(20) volume, followed by >2% up move within 3 candles.
# Bearish OB: candle with close < open and volume > 1.5x avg(20) volume, followed by >2% down move within 3 candles.
# Entry: Long when price retests OB low (within 0.5%) and closes above it with volume confirmation.
#        Short when price retests OB high (within 0.5%) and closes below it with volume confirmation.
# Uses 1d trend filter (price > EMA50 for long, < EMA50 for short) to avoid counter-trend trades.
# Designed for 6h timeframe with institutional footprint logic, works in both accumulation (bull) and distribution (bear) phases.

name = "6h_OrderBlock_Retest_Pattern"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-period average volume for volume spike detection
    vol_ma = np.zeros(n)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma[i] = vol_sum / 20.0
        else:
            vol_ma[i] = vol_sum / (i+1) if i+1 > 0 else 0.0
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_spike = volume > (1.5 * vol_ma)
    
    # Calculate 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Detect bullish and bearish order blocks
    bull_ob_low = np.full(n, np.nan)   # stores low of bullish OB
    bear_ob_high = np.full(n, np.nan)  # stores high of bearish OB
    
    for i in range(2, n-3):  # need 2 prior candles and 3 forward for validation
        # Bullish OB: bullish candle with volume spike, followed by >2% up move within 3 candles
        if (close[i] > open_price[i] and  # bullish candle
            vol_spike[i] and              # volume spike
            i+3 < n):
            # Check if any of next 3 candles achieve >2% gain from this candle's close
            future_max = np.max(high[i+1:i+4])
            if (future_max - close[i]) / close[i] > 0.02:  # >2% up move
                bull_ob_low[i] = low[i]  # mark the low as OB level
        
        # Bearish OB: bearish candle with volume spike, followed by >2% down move within 3 candles
        if (close[i] < open_price[i] and  # bearish candle
            vol_spike[i] and              # volume spike
            i+3 < n):
            # Check if any of next 3 candles achieve >2% drop from this candle's close
            future_min = np.min(low[i+1:i+4])
            if (close[i] - future_min) / close[i] > 0.02:  # >2% down move
                bear_ob_high[i] = high[i]  # mark the high as OB level
    
    # Forward-fill OB levels to make them available for retest (until invalidated)
    bull_ob_level = np.full(n, np.nan)
    bear_ob_level = np.full(n, np.nan)
    
    last_bull_ob = np.nan
    last_bear_ob = np.nan
    
    for i in range(n):
        if not np.isnan(bull_ob_low[i]):
            last_bull_ob = bull_ob_low[i]
        if not np.isnan(bear_ob_high[i]):
            last_bear_ob = bear_ob_high[i]
        
        bull_ob_level[i] = last_bull_ob
        bear_ob_level[i] = last_bear_ob
        
        # Invalidate OB if price breaks significantly beyond it
        if not np.isnan(last_bull_ob) and low[i] < last_bull_ob * 0.98:  # 2% break
            last_bull_ob = np.nan
        if not np.isnan(last_bear_ob) and high[i] > last_bear_ob * 1.02:  # 2% break
            last_bear_ob = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume average
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(bull_ob_level[i]) or 
            np.isnan(bear_ob_level[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price retests bullish OB (within 0.5%) and closes above it, with volume spike and above 1d EMA50
            if (bull_ob_level[i] > 0 and 
                abs(close[i] - bull_ob_level[i]) / bull_ob_level[i] <= 0.005 and  # within 0.5%
                close[i] > bull_ob_level[i] and 
                vol_spike[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price retests bearish OB (within 0.5%) and closes below it, with volume spike and below 1d EMA50
            elif (bear_ob_level[i] > 0 and 
                  abs(close[i] - bear_ob_level[i]) / bear_ob_level[i] <= 0.005 and  # within 0.5%
                  close[i] < bear_ob_level[i] and 
                  vol_spike[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below OB level or trend changes
            if close[i] < bull_ob_level[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above OB level or trend changes
            if close[i] > bear_ob_level[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals