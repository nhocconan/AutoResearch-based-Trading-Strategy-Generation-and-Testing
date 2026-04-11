#!/usr/bin/env python3
# 4h_12h_camarilla_pivot_volume_v1
# Strategy: 4h Camarilla pivot breakout with volume confirmation and 12h trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels act as strong support/resistance. Breakouts with volume
# confirmation indicate institutional participation. 12h EMA trend filter ensures trades align
# with higher timeframe momentum, reducing false breakouts. Designed for ~30-50 trades/year
# to minimize fee drag. Works in bull markets via breakout continuation and bear markets via
# short breakdowns during distribution phases.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 1d Camarilla pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_h4 = []
    camarilla_l4 = []
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_h4.append(np.nan)
            camarilla_l4.append(np.nan)
        else:
            # Camarilla levels: H4 = Close + 1.5 * (High - Low), L4 = Close - 1.5 * (High - Low)
            h4 = close_1d[i-1] + 1.5 * (high_1d[i-1] - low_1d[i-1])
            l4 = close_1d[i-1] - 1.5 * (high_1d[i-1] - low_1d[i-1])
            camarilla_h4.append(h4)
            camarilla_l4.append(l4)
    
    camarilla_h4 = np.array(camarilla_h4)
    camarilla_l4 = np.array(camarilla_l4)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or \
           np.isnan(camarilla_l4_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or \
           np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: price above/below 12h EMA50
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_h4_aligned[i]
        breakout_down = close[i] < camarilla_l4_aligned[i]
        
        # Entry conditions
        # Long: Bullish breakout above H4 + volume confirmation + price above 12h EMA50
        if breakout_up and vol_confirm and price_above_ema and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Bearish breakdown below L4 + volume confirmation + price below 12h EMA50
        elif breakout_down and vol_confirm and price_below_ema and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite breakout (price returns to pivot range)
        elif position == 1 and close[i] < camarilla_l4_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > camarilla_h4_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals