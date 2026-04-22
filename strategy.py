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
    
    # Load weekly data for multi-timeframe analysis - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    close_weekly = df_weekly['close'].values
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Calculate daily ATR(14) for volatility filtering
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    tr1 = high_daily[1:] - low_daily[1:]
    tr2 = np.abs(high_daily[1:] - close_daily[:-1])
    tr3 = np.abs(low_daily[1:] - close_daily[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14_daily = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_daily_aligned = align_htf_to_ltf(prices, df_daily, atr14_daily)
    
    # Calculate daily RSI(14) for overbought/oversold
    delta = pd.Series(close_daily).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi14_daily = 100 - (100 / (1 + rs))
    rsi14_daily_aligned = align_htf_to_ltf(prices, df_daily, rsi14_daily.values)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema50_weekly_aligned[i]) or 
            np.isnan(atr14_daily_aligned[i]) or 
            np.isnan(rsi14_daily_aligned[i])):
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
            # Long: Weekly uptrend + daily oversold bounce
            if (close[i] > ema50_weekly_aligned[i] and 
                rsi14_daily_aligned[i] < 30):
                signals[i] = 0.25
                position = 1
            # Short: Weekly downtrend + daily overbought rejection
            elif (close[i] < ema50_weekly_aligned[i] and 
                  rsi14_daily_aligned[i] > 70):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI mean reversion or trend change
            if position == 1:
                if rsi14_daily_aligned[i] > 70 or close[i] < ema50_weekly_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if rsi14_daily_aligned[i] < 30 or close[i] > ema50_weekly_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_WeeklyTrend_DailyRSI"
timeframe = "1d"
leverage = 1.0