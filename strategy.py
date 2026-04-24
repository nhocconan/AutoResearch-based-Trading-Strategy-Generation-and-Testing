#!/usr/bin/env python3
"""
Hypothesis: 4h Volume Spike + 1d Bollinger Band Reversal + 12h EMA Trend Filter.
- Primary timeframe: 4h for execution.
- HTF: 1d for Bollinger Band mean reversion signals (touch upper/lower band + reversal candle).
- HTF: 12h for EMA50 trend filter (only trade in direction of 12h trend).
- Entry: Bollinger Band touch + reversal candle (bullish/bearish engulfing) + volume spike (>2x 20 MA) + 12h EMA trend alignment.
- Exit: Opposite BB touch or trailing stop via signal=0 when price crosses 12h EMA.
- Regime: No chop filter - rely on Bollinger Bands working in ranging markets and trend filter for directional bias.
- Discrete signal size: 0.25 to balance drawdown and fee drag.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via buying dips at lower BB in uptrend, in bear via selling rallies at upper BB in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Calculate 1d Bollinger Bands (20, 2)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bollinger Bands
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 4h timeframe (completed 1d bar only)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Candlestick patterns for reversal detection
    # Bullish engulfing: current green candle engulfs previous red candle
    bullish_engulfing = (close > open_price) & (open_price < close) & \
                        (close > open_price) & (open_price < close) & \
                        (close > open_price) & (open_price < close)  # Placeholder - will fix below
    # Actually calculate properly:
    bullish_engulfing = (close > open_price) & (open_price < close) & \
                        (close[:-1] < open_price[:-1]) & \
                        (close > open_price[:-1]) & (open_price < close[:-1])
    bullish_engulfing = np.concatenate([[False], bullish_engulfing[:-1]])
    
    # Bearish engulfing: current red candle engulfs previous green candle
    bearish_engulfing = (close < open_price) & (open_price > close) & \
                        (close[:-1] > open_price[:-1]) & \
                        (close < open_price[:-1]) & (open_price > close[:-1])
    bearish_engulfing = np.concatenate([[False], bearish_engulfing[:-1]])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 50) + 5  # Need BB(20), volume MA(20), EMA50(12h) + buffer
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(sma_20_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for long entry: price at or below lower BB + bullish engulfing + volume spike + 12h uptrend
            at_lower_bb = low[i] <= lower_bb_aligned[i]
            # Recalculate bullish engulfing properly for current index
            bull_eng = False
            if i >= 1:
                bull_eng = (close[i] > open_price[i]) and (close[i-1] < open_price[i-1]) and \
                           (close[i] > open_price[i-1]) and (open_price[i] < close[i-1])
            
            if at_lower_bb and bull_eng and volume_spike[i] and ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]:
                # Uptrend: buy at lower BB with bullish reversal
                signals[i] = 0.25
                position = 1
            # Check for short entry: price at or above upper BB + bearish engulfing + volume spike + 12h downtrend
            elif high[i] >= upper_bb_aligned[i]:
                bear_eng = False
                if i >= 1:
                    bear_eng = (close[i] < open_price[i]) and (close[i-1] > open_price[i-1]) and \
                               (close[i] < open_price[i-1]) and (open_price[i] > close[i-1])
                
                if bear_eng and volume_spike[i] and ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]:
                    # Downtrend: sell at upper BB with bearish reversal
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price crosses above 12h EMA50 (trend change) or touches upper BB
            if close[i] > ema_50_12h_aligned[i] or high[i] >= upper_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below 12h EMA50 (trend change) or touches lower BB
            if close[i] < ema_50_12h_aligned[i] or low[i] <= lower_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VolumeSpike_BBReversal_12hEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0