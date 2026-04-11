#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume Weighted Average Price (VWAP) deviation with 12h trend filter.
# Long when price > VWAP and rising VWAP slope, with 12h EMA(50) uptrend and volume expansion.
# Short when price < VWAP and falling VWAP slope, with 12h EMA(50) downtrend and volume expansion.
# VWAP resets daily, capturing institutional participation and mean reversion to fair value.
# Designed for 15-35 trades/year on 6h timeframe with focus on institutional flow.
# Volume filter ensures moves have conviction, reducing false signals in chop.
# 12h trend filter prevents counter-trading in strong trends.

name = "6h_12h_vwap_deviation_trend_v1"
timeframe = "6h"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate daily VWAP (resets at 00:00 UTC)
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Price * volume
    pv = typical_price * volume
    # Cumulative sums
    cum_pv = np.cumsum(pv)
    cum_volume = np.cumsum(volume)
    # VWAP = cumulative PV / cumulative volume
    vwap = np.divide(cum_pv, cum_volume, out=np.zeros_like(cum_pv), where=cum_volume!=0)
    # Reset VWAP at daily boundary (00:00 UTC)
    dates = pd.to_datetime(prices['open_time']).date
    # Find where date changes
    date_changes = np.concatenate(([True], dates[1:] != dates[:-1]))
    # Reset cumulative sums at each new day
    cum_pv = np.where(date_changes, pv, cum_pv + pv)
    cum_volume = np.where(date_changes, volume, cum_volume + volume)
    vwap = np.divide(cum_pv, cum_volume, out=np.zeros_like(cum_pv), where=cum_volume!=0)
    
    # Calculate VWAP slope (3-period change)
    vwap_slope = vwap - np.roll(vwap, 3)
    # Handle first 3 values
    vwap_slope[:3] = 0
    
    # Calculate price deviation from VWAP as percentage
    price_dev = (close - vwap) / vwap * 100
    
    # Calculate volume moving average (20-period)
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after VWAP and volume MA warmup
        # Skip if any required data is invalid
        if (np.isnan(vwap[i]) or np.isnan(vwap_slope[i]) or 
            np.isnan(price_dev[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * 20-period average volume
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Determine 12h trend direction
        is_uptrend = close[i] > ema_50_12h_aligned[i]
        is_downtrend = close[i] < ema_50_12h_aligned[i]
        
        # VWAP deviation conditions
        price_above_vwap = price_dev[i] > 0.15  # Price > VWAP by 0.15%
        price_below_vwap = price_dev[i] < -0.15  # Price < VWAP by 0.15%
        vwap_rising = vwap_slope[i] > 0  # VWAP slope positive
        vwap_falling = vwap_slope[i] < 0  # VWAP slope negative
        
        # Entry conditions
        bullish_entry = price_above_vwap and vwap_rising and vol_filter and is_uptrend
        bearish_entry = price_below_vwap and vwap_falling and vol_filter and is_downtrend
        
        # Exit conditions: price crosses VWAP in opposite direction
        exit_long = price_dev[i] < -0.05  # Price crosses below VWAP by 0.05%
        exit_short = price_dev[i] > 0.05   # Price crosses above VWAP by 0.05%
        
        # Priority: entry > exit > hold
        if bullish_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_entry and position != -1:
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