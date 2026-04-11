#!/usr/bin/env python3
# 1h_4h_1d_camarilla_breakout_v1
# Strategy: 1h Camarilla breakout with 4h EMA trend and 1d volume confirmation
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: Camarilla levels provide institutional support/resistance. 4h EMA confirms trend direction.
# 1d volume filter ensures institutional participation. Designed for 15-35 trades/year to minimize fee drag.
# Works in bull markets via long breakouts above H3 and bear markets via short breakdowns below L3.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 30 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels for 1h (using previous candle)
    # Camarilla: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    # H2 = close + 1.1*(high-low)/6, L2 = close - 1.1*(high-low)/6
    # H1 = close + 1.1*(high-low)/12, L1 = close - 1.1*(high-low)/12
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    hl_range = prev_high - prev_low
    camarilla_h3 = prev_close + 1.1 * hl_range / 4
    camarilla_l3 = prev_close - 1.1 * hl_range / 4
    camarilla_h4 = prev_close + 1.1 * hl_range / 2
    camarilla_l4 = prev_close - 1.1 * hl_range / 2
    
    # 4h EMA20 for trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d volume average (20-period) for confirmation
    vol_avg_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Volume confirmation: current volume > 1.3x 1d average
        vol_confirm = volume[i] > 1.3 * vol_avg_20_1d_aligned[i]
        
        # Camarilla breakout signals
        breakout_h3 = high[i] > camarilla_h3[i]
        breakdown_l3 = low[i] < camarilla_l3[i]
        
        # 4h EMA trend filter
        trend_bullish = close[i] > ema_20_4h_aligned[i]
        trend_bearish = close[i] < ema_20_4h_aligned[i]
        
        # Entry conditions
        # Long: Break above H3 AND bullish trend AND volume confirmation
        if breakout_h3 and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.20
        # Short: Break below L3 AND bearish trend AND volume confirmation
        elif breakdown_l3 and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit: Opposite Camarilla level (L3 for long, H3 for short)
        elif position == 1 and low[i] < camarilla_l3[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and high[i] > camarilla_h3[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals