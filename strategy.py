#!/usr/bin/env python3
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
    
    # Load weekly data for 10-week EMA - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate EMA(10) from weekly close
    close_weekly = df_weekly['close'].values
    ema_10_weekly = pd.Series(close_weekly).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_10_weekly)
    
    # Load daily data for RSI(14) - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    # Calculate RSI(14) from daily close
    close_daily = df_daily['close'].values
    delta = np.diff(close_daily, prepend=close_daily[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_daily = 100 - (100 / (1 + rs))
    rsi_14_daily_aligned = align_htf_to_ltf(prices, df_daily, rsi_14_daily)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema_10_weekly_aligned[i]) or np.isnan(rsi_14_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above weekly EMA(10) and RSI < 30 (oversold)
            if (close[i] > ema_10_weekly_aligned[i] and 
                rsi_14_daily_aligned[i] < 30):
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly EMA(10) and RSI > 70 (overbought)
            elif (close[i] < ema_10_weekly_aligned[i] and 
                  rsi_14_daily_aligned[i] > 70):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60)
            if position == 1:
                if rsi_14_daily_aligned[i] > 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if rsi_14_daily_aligned[i] < 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_WeeklyEMA10_DailyRSI_MeanReversion"
timeframe = "1d"
leverage = 1.0