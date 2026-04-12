#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1h strategy using 4h Camarilla pivot breakouts with 1d EMA200 trend filter
    # 4h Camarilla levels provide intraday support/resistance structure
    # 1d EMA200 ensures we only trade in direction of daily trend (works in bull/bear)
    # Session filter (08-20 UTC) reduces noise during low-liquidity hours
    # Discrete sizing 0.20 to minimize fee churn. Target: 15-30 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla levels (using previous 4h bar)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    for i in range(len(prices)):
        # Find the index of the last completed 4h bar
        # Each 4h bar = 16 * 1h bars
        completed_4h_idx = (i // 16) * 16
        if completed_4h_idx >= 16 and completed_4h_idx < len(prices):
            # Use the completed 4h bar (index completed_4h_idx-16 to completed_4h_idx)
            start_idx = completed_4h_idx - 16
            end_idx = completed_4h_idx
            if start_idx >= 0 and end_idx <= len(high_4h):
                # Get the corresponding 4h bar data
                h4h_idx = start_idx // 16
                if h4h_idx < len(high_4h):
                    h4h_high = high_4h[h4h_idx]
                    h4h_low = low_4h[h4h_idx]
                    h4h_close = close_4h[h4h_idx]
                    pivot = (h4h_high + h4h_low + h4h_close) / 3
                    range_ = h4h_high - h4h_low
                    camarilla_h3[i] = pivot + range_ * 1.1 / 4
                    camarilla_l3[i] = pivot - range_ * 1.1 / 4
                    camarilla_h4[i] = pivot + range_ * 1.1 / 2
                    camarilla_l4[i] = pivot - range_ * 1.1 / 2
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Determine 1d trend
        bullish_trend = close[i] > ema200_1d_aligned[i]
        bearish_trend = close[i] < ema200_1d_aligned[i]
        
        # Entry logic: Camarilla breakout with trend filter
        long_entry = False
        short_entry = False
        
        # Long breakout: price breaks above H3 in bullish trend
        if bullish_trend:
            long_entry = (close[i] > camarilla_h3[i]) and (close[i-1] <= camarilla_h3[i-1] if i > 0 else False)
        # Short breakout: price breaks below L3 in bearish trend
        elif bearish_trend:
            short_entry = (close[i] < camarilla_l3[i]) and (close[i-1] >= camarilla_l3[i-1] if i > 0 else False)
        
        # Exit logic: opposite Camarilla level or trend reversal
        long_exit = (bearish_trend and close[i] < camarilla_l3[i]) or \
                   (not bullish_trend and not bearish_trend) or \
                   (close[i] < camarilla_l4[i])
        short_exit = (bullish_trend and close[i] > camarilla_h3[i]) or \
                    (not bullish_trend and not bearish_trend) or \
                    (close[i] > camarilla_h4[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_camarilla_pivot_breakout_trend_v1"
timeframe = "1h"
leverage = 1.0