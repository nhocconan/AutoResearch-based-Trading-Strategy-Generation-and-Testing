#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ConnorsRSI_TrendFilter_Weekly"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate RSI(3) - short-term RSI
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/3, adjust=False, min_periods=3).mean()
    avg_loss = loss.ewm(alpha=1/3, adjust=False, min_periods=3).mean()
    rs = avg_gain / avg_loss
    rsi3 = 100 - (100 / (1 + rs))
    rsi3 = rsi3.fillna(50).values
    
    # Calculate RSI of streak length
    up_days = np.zeros_like(close)
    down_days = np.zeros_like(close)
    for i in range(1, n):
        if close[i] > close[i-1]:
            up_days[i] = up_days[i-1] + 1
            down_days[i] = 0
        elif close[i] < close[i-1]:
            down_days[i] = down_days[i-1] + 1
            up_days[i] = 0
        else:
            up_days[i] = 0
            down_days[i] = 0
    
    # RSI of streak (using 2-period lookback as per Connors)
    streak_rsi = np.zeros(n)
    for i in range(n):
        if up_days[i] > 0:
            streak_val = min(up_days[i], 2)  # Cap at 2 for RSI calculation
        elif down_days[i] > 0:
            streak_val = min(down_days[i], 2)
        else:
            streak_val = 0
        
        # Calculate RSI for streak values
        if i >= 2:
            streak_slice = streak_val if isinstance(streak_val, np.ndarray) else [streak_val]
            # Simplified: use 0, 1, 2 levels for streak RSI
            if streak_val == 0:
                streak_rsi[i] = 50
            elif streak_val == 1:
                streak_rsi[i] = 30  # Oversold bias for up streak
            else:  # streak_val == 2
                streak_rsi[i] = 70  # Overbought bias for up streak
    
    # Percent Rank of close over last 100 periods
    percent_rank = np.zeros(n)
    lookback = 100
    for i in range(lookback, n):
        window = close[i-lookback:i]
        percent_rank[i] = (np.sum(window < close[i]) / len(window)) * 100
    # Fill beginning with 50
    percent_rank[:lookback] = 50
    
    # Connors RSI = (RSI(3) + RSI(Streak) + PercentRank(100)) / 3
    crsi = (rsi3 + streak_rsi + percent_rank) / 3
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: above 1.5x 24-period average (24*6h = 6 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 24)  # Wait for percent rank and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(crsi[i]) or np.isnan(ema_34_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]  # Volume confirmation
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long: CRSI < 15 (oversold) and price above weekly EMA (uptrend)
            if (crsi[i] < 15 and 
                close[i] > ema_34_6h[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: CRSI > 85 (overbought) and price below weekly EMA (downtrend)
            elif (crsi[i] > 85 and 
                  close[i] < ema_34_6h[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: CRSI > 70 (overbought) or price below weekly EMA
            if (crsi[i] > 70 or close[i] < ema_34_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: CRSI < 30 (oversold) or price above weekly EMA
            if (crsi[i] < 30 or close[i] > ema_34_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals