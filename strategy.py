#!/usr/bin/env python3
# 6h_1d_1w_camarilla_breakout_v1
# Strategy: 6-hour Camarilla pivot breakout with weekly filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels act as strong support/resistance. Breakouts above R3 or below S3
# with volume confirmation and aligned with weekly trend capture strong moves. Weekly trend filter
# avoids counter-trend trades in ranging markets. Designed for 15-35 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_camarilla_breakout_v1"
timeframe = "6h"
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
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Using previous day's OHLC to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    camarilla_r3 = prev_close + range_ * 1.1 / 2
    camarilla_s3 = prev_close - range_ * 1.1 / 2
    camarilla_r4 = prev_close + range_ * 1.1
    camarilla_s4 = prev_close - range_ * 1.1
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Weekly trend: price above/below weekly VWAP
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap = (typical_price * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    vwap_values = vwap.values
    weekly_vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap_values)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan( camarilla_r3_aligned[i] ) or np.isnan( camarilla_s3_aligned[i] ) or \
           np.isnan( camarilla_r4_aligned[i] ) or np.isnan( camarilla_s4_aligned[i] ) or \
           np.isnan( weekly_vwap_aligned[i] ) or np.isnan( vol_avg_20[i] ):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Weekly trend filter
        weekly_uptrend = close[i] > weekly_vwap_aligned[i]
        weekly_downtrend = close[i] < weekly_vwap_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_r3_aligned[i] and vol_confirm and weekly_uptrend
        breakdown_short = close[i] < camarilla_s3_aligned[i] and vol_confirm and weekly_downtrend
        
        # Exit conditions: reversal at opposite levels
        exit_long = close[i] < camarilla_s3_aligned[i]
        exit_short = close[i] > camarilla_r3_aligned[i]
        
        # Entry/exit logic
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakdown_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals