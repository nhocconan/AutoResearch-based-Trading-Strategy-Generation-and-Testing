#!/usr/bin/env python3
# 6h_1w_1d_volume_weighted_price_action_v1
# Hypothesis: Combines weekly volatility breakout with daily volume-weighted price action on 6h timeframe.
# Long when: price breaks above weekly high AND daily VWAP > previous close AND volume > 1.5x average
# Short when: price breaks below weekly low AND daily VWAP < previous close AND volume > 1.5x average
# Uses weekly range for structure and daily VWAP for intraday bias, reducing false breakouts.
# Targets 15-30 trades/year per symbol to avoid fee drag in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_volume_weighted_price_action_v1"
timeframe = "6h"
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
    
    # Weekly data for range breakout
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Daily data for VWAP and volume context
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate weekly high/low (use previous week to avoid look-ahead)
    prev_weekly_high = np.roll(high_1w, 1)
    prev_weekly_low = np.roll(low_1w, 1)
    prev_weekly_high[0] = np.nan
    prev_weekly_low[0] = np.nan
    
    # Calculate daily VWAP (typical price * volume / cumulative volume)
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = typical_price * volume_1d
    vwap_denominator = np.cumsum(volume_1d)
    vwap = np.where(vwap_denominator > 0, vwap_numerator / vwap_denominator, np.nan)
    # Reset VWAP at start of each day (simplified: use expanding window within day)
    # For simplicity, we use a 24-period rolling VWAP approximation on 1d data
    vwap_approx = pd.Series(typical_price).rolling(window=24, min_periods=24).mean().values
    
    # Previous day close for bias
    prev_daily_close = np.roll(close_1d, 1)
    prev_daily_close[0] = np.nan
    
    # Align weekly and daily data to 6h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, prev_weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, prev_weekly_low)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_approx)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_daily_close)
    
    # 6h volume average (24-period = 4 days)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or \
           np.isnan(vwap_aligned[i]) or np.isnan(prev_close_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * avg_volume[i]
        
        # Price relative to VWAP and previous close
        vwap_bias = close[i] > vwap_aligned[i]  # Above VWAP = bullish bias
        close_vs_prev_close = close[i] > prev_close_aligned[i]  # Above prev day close = bullish
        
        if position == 1:  # Long position
            # Exit: break below weekly low or loss of bullish bias
            if close[i] < weekly_low_aligned[i] or not vwap_bias:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: break above weekly high or loss of bearish bias
            if close[i] > weekly_high_aligned[i] or vwap_bias:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: break above weekly high with bullish VWAP bias
                if close[i] > weekly_high_aligned[i] and vwap_bias and close_vs_prev_close:
                    position = 1
                    signals[i] = 0.25
                # Short entry: break below weekly low with bearish VWAP bias
                elif close[i] < weekly_low_aligned[i] and not vwap_bias and not close_vs_prev_close:
                    position = -1
                    signals[i] = -0.25
    
    return signals