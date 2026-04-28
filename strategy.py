#!/usr/bin/env python3
"""
4h_RSI_Extreme_Trend_Filter_Volume
Hypothesis: Uses RSI(14) extremes (<20 for long, >80 for short) with 4h EMA50 trend filter and volume spike confirmation.
Designed to capture oversold/overbought reversals in both bull and bear markets by combining mean reversion with trend alignment.
Volume spike filters false signals. Targets 20-50 trades per year to minimize fee drag while capturing high-probability reversals.
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
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(14) on 4h closes
    delta = pd.Series(close_4h).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi_values)
    
    # Volume spike (>1.5x 20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from 4h EMA50
        trend_up = close[i] > ema_50_4h_aligned[i]
        trend_down = close[i] < ema_50_4h_aligned[i]
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # RSI extremes
        rsi_oversold = rsi_aligned[i] < 20
        rsi_overbought = rsi_aligned[i] > 80
        
        # Entry logic:
        # Long: Oversold in uptrend OR overbought reversal in downtrend (contrarian)
        long_entry = vol_confirm and (
            (rsi_oversold and trend_up) or  # Oversold bounce in uptrend
            (rsi_overbought and not trend_up)  # Overbought reversal in downtrend/sideways
        )
        
        # Short: Overbought in downtrend OR oversold reversal in uptrend (contrarian)
        short_entry = vol_confirm and (
            (rsi_overbought and trend_down) or  # Overbought rejection in downtrend
            (rsi_oversold and not trend_down)   # Oversold reversal in uptrend/sideways
        )
        
        # Exit logic: RSI returns to neutral zone (40-60) or trend reversal
        long_exit = (rsi_aligned[i] > 40 and not trend_up) or (rsi_aligned[i] > 60)
        short_exit = (rsi_aligned[i] < 60 and not trend_down) or (rsi_aligned[i] < 40)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_RSI_Extreme_Trend_Filter_Volume"
timeframe = "4h"
leverage = 1.0