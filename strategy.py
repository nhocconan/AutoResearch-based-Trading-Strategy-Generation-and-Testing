#!/usr/bin/env python3
"""
1h Camarilla H3/L3 Breakout with 4h EMA50 Trend Filter and Volume Spike Confirmation
Hypothesis: Camarilla pivot levels H3/L3 act as strong intraday resistance/support on 1h charts. 
A breakout above H3 or below L3 with 4h EMA50 trend alignment and volume spike (>2.0x 20-bar vol MA) 
captures strong momentum. Uses 4h/1d for signal direction, 1h only for entry timing to reduce trades.
Session filter (08-20 UTC) avoids low-liquidity periods. Discrete sizing (0.20) limits fee drag.
Target: 15-37 trades/year (60-150 over 4 years) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 51:  # Need 50 for EMA + 1 for shift
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = pd.Series(df_4h['close'])
    ema_50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 4h bar (using 4h data)
    # Camarilla: H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    prev_close_4h = df_4h['close'].shift(1).values
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    
    camarilla_h3_4h = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 4
    camarilla_l3_4h = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3_4h)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3_4h)
    
    # Calculate 20-period volume MA for volume spike confirmation (1h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA50, Camarilla, and volume MA
    start_idx = max(51, 20)  # 51 for EMA50 (50 + 1 for shift), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if outside trading session or data not ready
        if not in_session[i] or \
           (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_4h_aligned[i]
        h3_val = camarilla_h3_aligned[i]
        l3_val = camarilla_l3_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Trend filter: price above/below 4h EMA50
        price_above_ema = curr_close > ema_50_val
        price_below_ema = curr_close < ema_50_val
        
        if position == 0:
            # Long: break above H3 + price above 4h EMA50 + volume confirmation
            long_signal = (curr_high > h3_val) and price_above_ema and volume_confirm
            # Short: break below L3 + price below 4h EMA50 + volume confirmation
            short_signal = (curr_low < l3_val) and price_below_ema and volume_confirm
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses back below H3 OR price crosses below 4h EMA50
            if (curr_close < h3_val) or (curr_close < ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses back above L3 OR price crosses above 4h EMA50
            if (curr_close > l3_val) or (curr_close > ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0