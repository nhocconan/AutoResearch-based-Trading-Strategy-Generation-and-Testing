# 6h_OrderBlock_Breaker_1dTrend_Volume
# Hypothesis: Order blocks (last opposing candle before strong move) act as support/resistance.
# In trending markets (1d EMA50), price often revisits these blocks before continuing.
# Volume spike confirms institutional interest. Works in both bull/bear as it follows trend.
# Target: 50-150 trades over 4 years (12-37/year) on 6h timeframe.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_OrderBlock_Breaker_1dTrend_Volume"
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
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Identify bullish and bearish order blocks
    # Bullish OB: last red candle before a strong upward move (3+ consecutive green candles)
    # Bearish OB: last green candle before a strong downward move (3+ consecutive red candles)
    bullish_ob_high = np.full(n, np.nan)
    bullish_ob_low = np.full(n, np.nan)
    bearish_ob_high = np.full(n, np.nan)
    bearish_ob_low = np.full(n, np.nan)
    
    # Scan for order blocks
    for i in range(2, n-2):  # Need room to look ahead
        # Bullish OB: red candle (close < open) followed by 3+ green candles
        if (close[i] < prices['open'].iloc[i] and  # Red candle
            close[i+1] > prices['open'].iloc[i+1] and  # Next green
            close[i+2] > prices['open'].iloc[i+2] and
            close[i+3] > prices['open'].iloc[i+3]):
            bullish_ob_high[i] = high[i]  # OB high
            bullish_ob_low[i] = low[i]    # OB low
        
        # Bearish OB: green candle (close > open) followed by 3+ red candles
        if (close[i] > prices['open'].iloc[i] and  # Green candle
            close[i+1] < prices['open'].iloc[i+1] and  # Next red
            close[i+2] < prices['open'].iloc[i+2] and
            close[i+3] < prices['open'].iloc[i+3]):
            bearish_ob_high[i] = high[i]  # OB high
            bearish_ob_low[i] = low[i]    # OB low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price revisits bullish OB + uptrend + volume spike
            if not np.isnan(bullish_ob_low[i]) and not np.isnan(bullish_ob_high[i]):
                ob_low = bullish_ob_low[i]
                ob_high = bullish_ob_high[i]
                # Price in or near OB zone (within 0.5% of OB)
                near_ob = (low[i] <= ob_high * 1.005) and (high[i] >= ob_low * 0.995)
                if near_ob and (close[i] > ema_50_1d_aligned[i]) and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
            
            # Short: price revisits bearish OB + downtrend + volume spike
            elif not np.isnan(bearish_ob_low[i]) and not np.isnan(bearish_ob_high[i]):
                ob_low = bearish_ob_low[i]
                ob_high = bearish_ob_high[i]
                near_ob = (low[i] <= ob_high * 1.005) and (high[i] >= ob_low * 0.995)
                if near_ob and (close[i] < ema_50_1d_aligned[i]) and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: trend reversal or OB failure
            if close[i] < ema_50_1d_aligned[i]:  # Trend broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or OB failure
            if close[i] > ema_50_1d_aligned[i]:  # Trend broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals