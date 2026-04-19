#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d RSI momentum and 1w trend filter.
# Uses 1d RSI(14) for momentum signals and 1w EMA50 for trend direction.
# Enters long when RSI crosses above 30 (oversold recovery) in uptrend (price > 1w EMA50).
# Enters short when RSI crosses below 70 (overbought rejection) in downtrend (price < 1w EMA50).
# Includes volume confirmation (volume > 1.5x 20-period average) to filter low-quality signals.
# Designed for low trade frequency (target: 15-35 trades/year) to minimize fee drag.
# Works in bull/bear by following higher timeframe trend while capturing mean-reversion swings.
name = "12h_1dRSI_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (00-23 UTC - 12h timeframe less sensitive to intraday sessions)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 0) & (hours <= 23)  # Always active for 12h
    
    # Get 1d data for RSI(14) (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Get 1w data for EMA50 trend (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: volume > 1.5 * 20-period average (using 12h data)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_14_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI crosses above 30 (oversold recovery) AND price above 1w EMA50 (uptrend) with volume
            if (rsi_14_aligned[i] > 30 and rsi_14_aligned[i-1] <= 30 and 
                close[i] > ema_50_1w_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI crosses below 70 (overbought rejection) AND price below 1w EMA50 (downtrend) with volume
            elif (rsi_14_aligned[i] < 70 and rsi_14_aligned[i-1] >= 70 and 
                  close[i] < ema_50_1w_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if RSI crosses below 50 (momentum loss) or price breaks below 1w EMA50
            if (rsi_14_aligned[i] < 50 and rsi_14_aligned[i-1] >= 50) or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if RSI crosses above 50 (momentum loss) or price breaks above 1w EMA50
            if (rsi_14_aligned[i] > 50 and rsi_14_aligned[i-1] <= 50) or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals