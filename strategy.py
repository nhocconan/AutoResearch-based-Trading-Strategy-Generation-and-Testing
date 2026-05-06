#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d RSI extremes with 1w EMA34 trend filter and volume confirmation
# Long when 1d RSI < 30 (oversold) AND 1w EMA34 > EMA34 previous (uptrend) AND volume > 1.8 * avg_volume(20) on 12h
# Short when 1d RSI > 70 (overbought) AND 1w EMA34 < EMA34 previous (downtrend) AND volume > 1.8 * avg_volume(20) on 12h
# Exit when 1d RSI crosses back through 50 (mean reversion to midpoint)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# RSI extremes provide high-probability reversal points in ranging markets
# 1w EMA34 trend filter ensures we trade with the dominant weekly trend
# Volume confirmation (1.8x) validates reversal strength while limiting overtrading
# Works in both bull (buy oversold dips) and bear (sell overbought rallies) markets

name = "12h_1dRSI_Extreme_1wEMA34_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least 14 completed 1d bars for RSI
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    # Handle division by zero (when avg_loss == 0)
    rsi_1d = np.where(avg_loss.values == 0, 100, rsi_1d)
    rsi_1d = np.where(np.isnan(rsi_1d), 50, rsi_1d)
    
    # Align 1d RSI to 12h timeframe (wait for completed 1d bar)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need at least 34 completed weekly bars for EMA34
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold), 1w EMA34 > EMA34 previous (uptrend), volume spike, in session
            if (rsi_aligned[i] < 30 and 
                ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought), 1w EMA34 < EMA34 previous (downtrend), volume spike, in session
            elif (rsi_aligned[i] > 70 and 
                  ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI crosses back above 50 (mean reversion)
            if rsi_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI crosses back below 50 (mean reversion)
            if rsi_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals