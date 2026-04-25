#!/usr/bin/env python3
"""
4h Camarilla R1/S1 Breakout + 1d EMA34 Trend + Volume Spike with ATR Stoploss
Hypothesis: Camarilla pivot levels from daily chart identify key institutional support/resistance.
Breakouts above R1 or below S1 with daily EMA34 trend alignment and volume spike capture
institutional participation. ATR-based stoploss limits drawdown. Works in bull markets (trend
continuation) and bear markets (mean reversion to pivot levels). 4h timeframe targets
20-50 trades/year to avoid fee drag.
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
    open_ = prices['open'].values
    
    # Daily data for Camarilla pivots and EMA34 (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for daily data
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    # R2 = close + 0.55*(high-low), R1 = close + 0.275*(high-low)
    # S1 = close - 0.275*(high-low), S2 = close - 0.55*(high-low), 
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    daily_range = daily_high - daily_low
    camarilla_r1 = daily_close + 0.275 * daily_range
    camarilla_s1 = daily_close - 0.275 * daily_range
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Daily EMA34 trend filter
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for daily data and volume MA
    start_idx = max(34, 20, 14) + 5  # extra for safety
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        
        # Breakout conditions
        breakout_long = curr_close > camarilla_r1_aligned[i]
        breakout_short = curr_close < camarilla_s1_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla breakout + volume spike + daily EMA34 trend alignment
            long_entry = breakout_long and vol_spike and (curr_close > ema_34_1d_aligned[i])
            short_entry = breakout_short and vol_spike and (curr_close < ema_34_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on retrace to S1, trend change, or ATR stoploss
            stoploss_level = entry_price - 2.5 * atr[i]
            if curr_close < camarilla_s1_aligned[i] or curr_close < ema_34_1d_aligned[i] or curr_close < stoploss_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on retrace to R1, trend change, or ATR stoploss
            stoploss_level = entry_price + 2.5 * atr[i]
            if curr_close > camarilla_r1_aligned[i] or curr_close > ema_34_1d_aligned[i] or curr_close > stoploss_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0