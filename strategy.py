#!/usr/bin/env python3
"""
12h Camarilla Pivot H3/L3 Breakout + 1d EMA34 Trend + Volume Spike Confirmation
Hypothesis: Camarilla pivot levels (H3/L3) act as strong intraday support/resistance.
Breakouts above H3 or below L3 with volume confirmation and 1d EMA34 trend filter capture strong moves.
In ranging markets (price between H3/L3), we stay flat to avoid false breakouts.
Uses 12h primary timeframe with 1d EMA34 for higher timeframe trend filter.
Designed for BTC/ETH with 50-150 total trades over 4 years to minimize fee drag while capturing strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation and EMA34 (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need 34 for EMA34
        return np.zeros(n)
    
    # Calculate 1d typical price for pivot points
    typical_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # H3 = close + 1.1 * (high - low) / 2
    # L3 = close - 1.1 * (high - low) / 2
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 24-period volume MA for volume spike confirmation (12h)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34 and volume MA
    start_idx = max(34, 24)  # 34 for EMA34, 24 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_ma = vol_ma_24[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        ranging = (curr_close >= l3_level) and (curr_close <= h3_level)  # price between H3/L3
        
        # Volume confirmation: current volume > 2.0 * 24-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            if ranging:
                # Market ranging between H3/L3: stay flat
                signals[i] = 0.0
                position = 0
            elif uptrend:
                # Uptrend: look for long when price breaks above H3 with volume
                long_signal = (curr_close > h3_level) and volume_confirm
                if long_signal:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
                    position = 0
            elif downtrend:
                # Downtrend: look for short when price breaks below L3 with volume
                short_signal = (curr_close < l3_level) and volume_confirm
                if short_signal:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
                    position = 0
        elif position == 1:
            # Exit long: price closes below H3 OR EMA34 trend turns down
            if curr_close <= h3_level or curr_close <= ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above L3 OR EMA34 trend turns up
            if curr_close >= l3_level or curr_close >= ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0