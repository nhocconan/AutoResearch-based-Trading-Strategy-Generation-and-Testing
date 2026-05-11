#!/usr/bin/env python3
"""
1D_RSI_Extremes_1wTrend_Filter
Hypothesis: Weekly EMA20 defines long-term trend. Daily RSI extremes (overbought/oversold) provide mean-reversion entries in the direction of the weekly trend. Volume spike confirms institutional interest. Designed for low turnover in ranging 2025 market, targeting 10-25 trades/year.
"""

name = "1D_RSI_Extremes_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0

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
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate daily RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: 20-day EMA for spike detection
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # Fixed position size to minimize churn
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_weekly_ema = close[i] > ema20_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema20_1w_aligned[i]
        rsi_oversold = rsi_values[i] < 30
        rsi_overbought = rsi_values[i] > 70
        
        if position == 0:
            # Long: RSI oversold + price above weekly EMA20 + volume spike
            if rsi_oversold and price_above_weekly_ema and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: RSI overbought + price below weekly EMA20 + volume spike
            elif rsi_overbought and price_below_weekly_ema and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60) or trend reversal
            if position == 1:
                if rsi_values[i] > 40 or close[i] < ema20_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if rsi_values[i] < 60 or close[i] > ema20_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals