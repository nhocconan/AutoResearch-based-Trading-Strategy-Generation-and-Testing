#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation (>2.0x 20-bar average) captures strong continuation moves. Uses discrete sizing (0.25) targeting ~12-37 trades/year on 6f. Works in bull/bear by only taking breakouts aligned with 1d trend. No stoploss - exit on opposite Camarilla touch (R4/S4) or time-based exit to reduce whipsaw.
"""

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
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 6h bar
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We use previous bar's OHLC to avoid look-ahead
    prev_close = np.concatenate([[np.nan], close[:-1]])
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    
    range_hl = prev_high - prev_low
    camarilla_r3 = prev_close + (range_hl * 1.1 / 4)
    camarilla_s3 = prev_close - (range_hl * 1.1 / 4)
    camarilla_r4 = prev_close + (range_hl * 1.1 / 2)
    camarilla_s4 = prev_close - (range_hl * 1.1 / 2)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of EMA34 (34), volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        trend_val = ema34_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Skip if any data not ready
        if (np.isnan(trend_val) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price > 1d EMA34 = uptrend, price < 1d EMA34 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Entry conditions: Camarilla R3/S3 breakout in direction of 1d trend + volume
        long_entry = (close_val > camarilla_r3[i]) and is_uptrend and vol_conf
        short_entry = (close_val < camarilla_s3[i]) and is_downtrend and vol_conf
        
        # Exit conditions: 
        # 1. Opposite Camarilla touch (R4/S4) - means breakout failed
        # 2. Camarilla center (mean reversion) - take profit at midpoint
        long_exit = False
        short_exit = False
        if position == 1:
            # Long exit: price touches R4 (failed breakout) or reaches midpoint (profit)
            long_exit = (close_val >= camarilla_r4[i]) or (close_val <= (camarilla_r3[i] + camarilla_s3[i]) / 2)
        elif position == -1:
            # Short exit: price touches S4 (failed breakdown) or reaches midpoint (profit)
            short_exit = (close_val <= camarilla_s4[i]) or (close_val >= (camarilla_r3[i] + camarilla_s3[i]) / 2)
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0