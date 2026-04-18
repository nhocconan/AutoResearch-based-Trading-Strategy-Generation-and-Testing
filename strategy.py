#!/usr/bin/env python3
"""
4h_HighLow_Reversal_Strategy_v1
Hypothesis: Price often reverses from recent highs/lows in 4h timeframe. 
Go long when price closes near 4h low with volume confirmation and RSI oversold.
Go short when price closes near 4h high with volume confirmation and RSI overbought.
Use daily trend filter to avoid counter-trend trades. 
Target: 20-40 trades/year by requiring confluence of price action, volume, and RSI extremes.
Works in both bull/bear markets via mean reversion at swing points and trend alignment filter.
"""

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
    
    # Calculate 4-period RSI for momentum
    def calculate_rsi(close_prices, period=4):
        if len(close_prices) < period + 1:
            return np.full_like(close_prices, np.nan)
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_prices, np.nan)
        avg_loss = np.full_like(close_prices, np.nan)
        
        if len(gain) >= period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            for i in range(period + 1, len(close_prices)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 4)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA(50) for trend filter
    if len(close_1d) >= 50:
        ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    else:
        ema_1d = np.full_like(close_1d, np.nan)
    
    # Align daily EMA to 4h timeframe
    ema_1d_4h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 4) + 1  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(ema_1d_4h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Price action: close near recent high/low
        lookback = 4
        if i >= lookback:
            recent_high = np.max(high[i-lookback:i])
            recent_low = np.min(low[i-lookback:i])
            range_ = recent_high - recent_low
            
            if range_ > 0:
                # Position of close within recent range (0 = at low, 1 = at high)
                close_position = (close[i] - recent_low) / range_
            else:
                close_position = 0.5
        else:
            close_position = 0.5
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter: only trade in direction of daily trend
        uptrend = close[i] > ema_1d_4h[i]
        downtrend = close[i] < ema_1d_4h[i]
        
        if position == 0:
            # Long: price near low, RSI oversold, volume confirmation, uptrend
            if (close_position < 0.3 and  # near recent low
                rsi[i] < 30 and          # oversold
                vol_confirm and 
                uptrend):
                signals[i] = 0.25
                position = 1
            # Short: price near high, RSI overbought, volume confirmation, downtrend
            elif (close_position > 0.7 and  # near recent high
                  rsi[i] > 70 and           # overbought
                  vol_confirm and 
                  downtrend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price near high OR RSI overbought
            if (close_position > 0.7 or rsi[i] > 70):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price near low OR RSI oversold
            if (close_position < 0.3 or rsi[i] < 30):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HighLow_Reversal_Strategy_v1"
timeframe = "4h"
leverage = 1.0