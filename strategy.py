#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R extreme + 12h EMA200 trend filter + volume spike
    # Long: Williams %R(14) < -80 (oversold) AND price > 12h EMA200 AND volume > 1.5x 20-period average
    # Short: Williams %R(14) > -20 (overbought) AND price < 12h EMA200 AND volume > 1.5x 20-period average
    # Exit: Williams %R returns to -50 (mean reversion)
    # Using 12h for EMA200 (long-term trend) and Williams %R calculation (avoid look-ahead), 6h only for entry/exit timing
    # Discrete position sizing (0.25) to minimize fee churn and manage drawdown
    # Target: 12-25 trades/year (~50-100 over 4 years) to stay well below 300 max trades limit
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R and EMA200 (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R on 12h: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    williams_r = np.full(len(close_12h), np.nan)
    for i in range(14, len(close_12h)):
        highest_high = np.max(high_12h[i-14:i+1])
        lowest_low = np.min(low_12h[i-14:i+1])
        if highest_high != lowest_low:
            williams_r[i] = ((highest_high - close_12h[i]) / (highest_high - lowest_low)) * -100
        else:
            williams_r[i] = -50  # neutral if no range
    
    # 12h EMA200 for trend filter
    ema_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h indicators to 6h (wait for completed 12h bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: >1.5x 20-period average (to reduce false signals)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Williams %R extreme levels
        williams_r_val = williams_r_aligned[i]
        oversold = williams_r_val < -80
        overbought = williams_r_val > -20
        mean_reversion_exit = abs(williams_r_val + 50) < 10  # near -50
        
        # Trend filter: only long if price > 12h EMA200, only short if price < 12h EMA200
        long_trend_ok = close[i] > ema_12h_aligned[i]
        short_trend_ok = close[i] < ema_12h_aligned[i]
        
        # Entry logic: Williams %R extreme + volume + trend
        long_entry = oversold and vol_confirm and long_trend_ok
        short_entry = overbought and vol_confirm and short_trend_ok
        
        # Exit logic: Williams %R mean reversion (return to -50)
        long_exit = mean_reversion_exit
        short_exit = mean_reversion_exit
        
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

name = "6h_12h_williamsr_extreme_volume_trend_v1"
timeframe = "6h"
leverage = 1.0