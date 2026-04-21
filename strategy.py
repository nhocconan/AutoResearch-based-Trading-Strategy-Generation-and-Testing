#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly trend filter with daily RSI mean reversion and volume confirmation.
# In bull markets (weekly price > weekly 50-period SMA): buy RSI oversold (<30) on volume spike.
# In bear markets (weekly price < weekly 50-period SMA): sell RSI overbought (>70) on volume spike.
# Uses volume > 1.5x 20-period average for confirmation. Targets 15-25 trades/year.
# Weekly trend filter reduces whipsaws, RSI provides mean-reversion entries, volume confirms conviction.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Weekly 50-period SMA for trend filter
    weekly_close = df_weekly['close'].values
    weekly_sma50 = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    weekly_sma50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_sma50)
    
    # Daily RSI (14-period) for mean reversion
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/14)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = 50  # Initialize with neutral value
    
    # Daily volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(weekly_sma50_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Weekly trend filter
        weekly_price = df_weekly['close'].values
        # Find corresponding weekly index (simplified: use last known weekly value)
        weekly_idx = min(i // (24*7), len(weekly_price)-1)  # Approximate weekly bars
        weekly_trend_up = weekly_price[weekly_idx] > weekly_sma50[weekly_idx]
        
        if position == 0:
            if volume_confirm:
                # Bull market: buy RSI oversold
                if weekly_trend_up and rsi[i] < 30:
                    signals[i] = 0.25
                    position = 1
                # Bear market: sell RSI overbought
                elif not weekly_trend_up and rsi[i] > 70:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on RSI overbought or trend change
                if rsi[i] > 70 or not weekly_trend_up:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on RSI oversold or trend change
                if rsi[i] < 30 or weekly_trend_up:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyTrend_RSI_MeanRev_Volume"
timeframe = "1d"
leverage = 1.0