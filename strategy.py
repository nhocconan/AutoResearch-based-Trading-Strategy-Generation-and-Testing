#!/usr/bin/env python3
# 12h_1w_1d_camarilla_volume_crossover_v1
# Hypothesis: Use weekly Camarilla pivot levels from daily data for support/resistance, combined with daily trend filter and volume confirmation on 12h timeframe.
# Goes long when price breaks above daily H4 resistance with volume confirmation in bull market (price > weekly EMA20).
# Goes short when price breaks below daily L4 support with volume confirmation in bear market (price < weekly EMA20).
# Uses Camarilla's institutional levels for high-probability breakouts, volume filter to avoid false breaks, and weekly trend to avoid counter-trend trades.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) by requiring confluence of level break, volume, and trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_camarilla_volume_crossover_v1"
timeframe = "12h"
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
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L) [using previous day]
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align Camarilla levels to 12h timeframe (already delayed by shift(1))
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Weekly EMA20 for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    
    # Volume confirmation: volume > 1.8x average of last 4 periods (2 days in 12h)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    vol_confirm = volume > vol_ma * 1.8
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or \
           np.isnan(weekly_ema20_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Determine market trend based on weekly EMA20
        bull_market = close[i] > weekly_ema20_aligned[i]
        bear_market = close[i] < weekly_ema20_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes back below H4 or trend turns bearish
            if close[i] < camarilla_h4_aligned[i] or bear_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes back above L4 or trend turns bullish
            if close[i] > camarilla_l4_aligned[i] or bull_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above H4 with volume in bull market
            if bull_market and close[i] > camarilla_h4_aligned[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below L4 with volume in bear market
            elif bear_market and close[i] < camarilla_l4_aligned[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals