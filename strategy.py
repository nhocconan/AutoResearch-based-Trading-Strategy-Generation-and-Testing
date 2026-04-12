#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d KAMA trend filter with 1w Williams %R mean reversion and volume confirmation
    # KAMA adapts to market efficiency - trending when ER high, mean-reverting when ER low
    # 1w Williams %R < -80 = oversold (long bias), > -20 = overbought (short bias) in 1w context
    # Volume > 1.3x 20-period MA confirms institutional participation
    # Discrete position sizing (0.25) to minimize fee churn. Target: 15-25 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Williams %R
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Williams %R (14-period)
    highest_14_1w = np.full(len(close_1w), np.nan)
    lowest_14_1w = np.full(len(close_1w), np.nan)
    
    for i in range(14, len(close_1w)):
        highest_14_1w[i] = np.max(high_1w[i-14:i])
        lowest_14_1w[i] = np.min(low_1w[i-14:i])
    
    williams_r_1w = np.full(len(close_1w), np.nan)
    for i in range(14, len(close_1w)):
        if highest_14_1w[i] != lowest_14_1w[i]:
            williams_r_1w[i] = (highest_14_1w[i] - close_1w[i]) / (highest_14_1w[i] - lowest_14_1w[i]) * -100
        else:
            williams_r_1w[i] = -50.0
    
    williams_r_1w_aligned = align_htf_to_ltf(prices, df_1w, williams_r_1w)
    
    # KAMA (Adaptive Moving Average) on 1d close
    # Efficiency Ratio (ER) = |net change| / sum(|changes|)
    # Smoothing Constants: fastest SC=2/(2+1)=0.667, slowest SC=2/(30+1)=0.0645
    close_series = pd.Series(close)
    change = abs(close_series.diff(1))
    volatility = change.rolling(window=10, min_periods=1).sum()
    net_change = abs(close_series.diff(10))
    er = np.where(volatility > 0, net_change / volatility, 0)
    
    # Smoothing constant
    sc = (er * (0.667 - 0.0645) + 0.0645) ** 2
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    
    for i in range(1, n):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    vol_ratio = np.full(n, np.nan)
    for i in range(20, n):
        if vol_ma_20[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma_20[i]
        else:
            vol_ratio[i] = 1.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(williams_r_1w_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from KAMA (price above/below KAMA)
        uptrend = close[i] > kama[i]
        downtrend = close[i] < kama[i]
        
        # 1w Williams %R mean reversion conditions
        oversold = williams_r_1w_aligned[i] < -80
        overbought = williams_r_1w_aligned[i] > -20
        
        # Entry conditions with volume confirmation
        long_entry = oversold and (vol_ratio[i] > 1.3) and uptrend
        short_entry = overbought and (vol_ratio[i] > 1.3) and downtrend
        
        # Exit conditions: Williams %R returns to midpoint (-50)
        long_exit = williams_r_1w_aligned[i] > -50
        short_exit = williams_r_1w_aligned[i] < -50
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_kama_williams_r_mean_reversion_vol_v1"
timeframe = "1d"
leverage = 1.0