#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly RSI Extreme + Volume Confirmation with 1d Trend Filter
# Uses weekly RSI extremes (overbought/oversold) to catch reversals
# 1d EMA (50) provides trend filter to avoid counter-trend trades
# Volume spike (>2x 20-period average) confirms momentum at reversal points
# Works in bull/bear markets by capturing exhaustion moves
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for RSI
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly RSI (14)
    close_weekly = df_weekly['close'].values
    delta = np.diff(close_weekly, prepend=close_weekly[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_weekly = 100 - (100 / (1 + rs))
    rsi_weekly_aligned = align_htf_to_ltf(prices, df_weekly, rsi_weekly)
    
    # Load daily data ONCE before loop for trend filter
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMA (50) for trend direction
    close_daily = df_daily['close'].values
    ema_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Volume spike detection: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 14)  # for volume MA and RSI
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_weekly_aligned[i]) or 
            np.isnan(ema_daily_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: only trade in direction of daily EMA
        above_ema = price > ema_daily_aligned[i]
        
        if position == 0:
            # Long: weekly RSI oversold (<30) + volume spike + uptrend filter
            if (rsi_weekly_aligned[i] < 30 and volume_spike[i] and above_ema):
                position = 1
                signals[i] = position_size
            # Short: weekly RSI overbought (>70) + volume spike + downtrend filter
            elif (rsi_weekly_aligned[i] > 70 and volume_spike[i] and not above_ema):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: weekly RSI returns to neutral (>50) or trend changes
            if rsi_weekly_aligned[i] > 50 or price < ema_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: weekly RSI returns to neutral (<50) or trend changes
            if rsi_weekly_aligned[i] < 50 or price > ema_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WeeklyRSI_Extreme_Volume_Spike"
timeframe = "6h"
leverage = 1.0