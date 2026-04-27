#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams %R reversal with 1-day trend filter and volume confirmation.
In bull market (price > 1-day EMA34): long when Williams %R crosses above -80 (oversold) and volume > 1.2x average.
In bear market (price < 1-day EMA34): short when Williams %R crosses below -20 (overbought) and volume > 1.2x average.
Williams %R identifies momentum exhaustion, daily trend filters direction, volume confirms institutional participation.
Target: 12-35 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend
    daily_close = df_1d['close'].values
    ema_34_1d = np.empty_like(daily_close, dtype=np.float64)
    ema_34_1d.fill(np.nan)
    if len(daily_close) >= 34:
        alpha = 2.0 / (34 + 1)
        ema_34_1d[33] = np.mean(daily_close[:34])
        for i in range(34, len(daily_close)):
            ema_34_1d[i] = alpha * daily_close[i] + (1 - alpha) * ema_34_1d[i-1]
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Williams %R (14-period)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close_wr = df_1d['close'].values
    
    highest_high = np.full_like(daily_high, np.nan)
    lowest_low = np.full_like(daily_low, np.nan)
    for i in range(13, len(daily_high)):
        highest_high[i] = np.max(daily_high[i-13:i+1])
        lowest_low[i] = np.min(daily_low[i-13:i+1])
    
    williams_r = np.full_like(daily_close, np.nan)
    for i in range(13, len(daily_close)):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = -100 * (highest_high[i] - daily_close_wr[i]) / (highest_high[i] - lowest_low[i])
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get daily data for volume confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = np.empty_like(vol_1d, dtype=np.float64)
    vol_ma_20_1d.fill(np.nan)
    for i in range(19, len(vol_1d)):
        vol_ma_20_1d[i] = np.mean(vol_1d[i-19:i+1])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Williams %R (14), daily EMA34 (34), daily volume MA20 (20)
    start_idx = max(14, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        wr_now = williams_r_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Daily close price for trend comparison
        daily_close_price = df_1d['close'].values
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close_price)
        if np.isnan(daily_close_aligned[i]):
            signals[i] = 0.0
            continue
        daily_close_val = daily_close_aligned[i]
        
        # Volume filter: volume > 1.2x daily average
        vol_filter = vol_now > 1.2 * vol_ma
        
        # Williams %R signals with crossover detection
        if i > start_idx:
            wr_prev = williams_r_aligned[i-1]
            wr_cross_above_80 = (wr_prev <= -80) and (wr_now > -80)  # Oversold bounce
            wr_cross_below_20 = (wr_prev >= -20) and (wr_now < -20)  # Overbought rejection
        else:
            wr_cross_above_80 = False
            wr_cross_below_20 = False
        
        if position == 0:
            # Bull market (price > daily EMA34): look for long when WR crosses above -80
            if daily_close_val > ema_trend and wr_cross_above_80 and vol_filter:
                signals[i] = size
                position = 1
            # Bear market (price < daily EMA34): look for short when WR crosses below -20
            elif daily_close_val < ema_trend and wr_cross_below_20 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: WR crosses above -20 (overbought) or trend changes to bear
            if wr_now >= -20 or daily_close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: WR crosses below -80 (oversold) or trend changes to bull
            if wr_now <= -80 or daily_close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsR_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0