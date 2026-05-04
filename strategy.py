#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter (EMA50) and volume confirmation (>1.5x 20 EMA volume)
# Uses 1h RSI for mean reversion entries - long when RSI<30, short when RSI>70
# 4h EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation filters low-momentum breakouts (>1.5x average volume)
# Session filter (08-20 UTC) reduces noise trades during low-volatility periods
# Discrete sizing 0.20 minimizes fee churn while maintaining profitability
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# Works in bull markets (mean reversion in uptrend) and bear markets (mean reversion in downtrend)
# Focus on BTC/ETH by requiring 4h trend alignment (avoids SOL-only bias)

name = "1h_RSI14_4hEMA50_VolumeConfirm_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(50) trend filter from prior completed 4h bar
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_shifted = np.roll(ema_50_4h, 1)
    ema_50_4h_shifted[0] = np.nan
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(rsi_values[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: RSI < 30 (oversold) AND price > 4h EMA50 AND volume spike
            if rsi_values[i] < 30 and close[i] > ema_50_4h_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: RSI > 70 (overbought) AND price < 4h EMA50 AND volume spike
            elif rsi_values[i] > 70 and close[i] < ema_50_4h_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI returns to 50 (mean reversion) OR price crosses below 4h EMA50
            if rsi_values[i] >= 50 or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI returns to 50 (mean reversion) OR price crosses above 4h EMA50
            if rsi_values[i] <= 50 or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals