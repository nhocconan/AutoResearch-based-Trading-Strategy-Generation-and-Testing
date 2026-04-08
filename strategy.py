#!/usr/bin/env python3
# 1d_weekly_momentum_reversal_v1
# Hypothesis: Weekly momentum reversal with volume confirmation and daily trend filter.
# Buy when weekly RSI < 30 (oversold) and price closes above weekly open with volume > 1.5x average.
# Sell when weekly RSI > 70 (overbought) and price closes below weekly open with volume > 1.5x average.
# Exit when weekly RSI returns to neutral zone (40-60) or opposite signal.
# Designed to work in both bull and bear markets by capturing mean reversion in extreme conditions.
# Target: 15-25 trades/year to minimize fee decay while capturing high-probability reversals.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_momentum_reversal_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get weekly data for RSI calculation (calculate once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly RSI(14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly open price
    open_1w = df_1w['open'].values
    
    # Align weekly RSI and open to daily chart
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    open_1w_aligned = align_htf_to_ltf(prices, df_1w, open_1w)
    
    # Daily trend: close > open = uptrend, close < open = downtrend
    daily_uptrend = close > open_price
    daily_downtrend = close < open_price
    
    # Volume confirmation: 20-day average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(rsi_aligned[i]) or np.isnan(open_1w_aligned[i]) or \
           np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral zone or opposite signal
            if rsi_aligned[i] >= 40 or \
               (close[i] < open_1w_aligned[i] and volume[i] > 1.5 * avg_volume[i] and daily_downtrend[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral zone or opposite signal
            if rsi_aligned[i] <= 60 or \
               (close[i] > open_1w_aligned[i] and volume[i] > 1.5 * avg_volume[i] and daily_uptrend[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: weekly RSI < 30 (oversold) and price closes above weekly open with volume and daily uptrend
            if rsi_aligned[i] < 30 and close[i] > open_1w_aligned[i] and volume_ok and daily_uptrend[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: weekly RSI > 70 (overbought) and price closes below weekly open with volume and daily downtrend
            elif rsi_aligned[i] > 70 and close[i] < open_1w_aligned[i] and volume_ok and daily_downtrend[i]:
                position = -1
                signals[i] = -0.25
    
    return signals