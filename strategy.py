#!/usr/bin/env python3
"""
6h_RSI2_Recovery_1dTrend_Volume_v1
Hypothesis: Trade short-term RSI(2) mean reversion on 6h timeframe aligned with 1-day trend and volume confirmation.
In bull markets: buy when RSI(2) < 10 (extreme oversold) and price > daily EMA50 with volume > 1.5x average.
In bear markets: sell when RSI(2) > 90 (extreme overbought) and price < daily EMA50 with volume > 1.5x average.
Exit on RSI(2) crossing 50 (mean reversion complete) or trend reversal.
RSI(2) captures very short-term exhaustion, daily EMA50 filters trend direction, volume confirms conviction.
Position size: 0.25 to limit drawdown. Target: 50-150 total trades over 4 years = 12-37/year.
Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for confirmation (using 1d volume)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate RSI(2) on 6h close
    def rsi(close_vals, period):
        delta = np.diff(close_vals)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close_vals)
        avg_loss = np.zeros_like(close_vals)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period + 1, len(close_vals)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        rsi_vals[:period] = np.nan
        return rsi_vals
    
    rsi_2 = rsi(close, 2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for RSI(2) (2) and EMA50 (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_2[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above daily EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long setup: RSI(2) < 10 (extreme oversold) + 1d uptrend + volume confirmation
            long_setup = (rsi_2[i] < 10) and htf_1d_bullish and volume_confirm
            
            # Short setup: RSI(2) > 90 (extreme overbought) + 1d downtrend + volume confirmation
            short_setup = (rsi_2[i] > 90) and htf_1d_bearish and volume_confirm
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: RSI(2) crosses above 50 (mean reversion) OR 1d trend turns bearish
            if (rsi_2[i] > 50) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: RSI(2) crosses below 50 (mean reversion) OR 1d trend turns bullish
            if (rsi_2[i] < 50) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_RSI2_Recovery_1dTrend_Volume_v1"
timeframe = "6h"
leverage = 1.0