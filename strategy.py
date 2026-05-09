#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot (R1/S1) breakout with 1d VWAP trend filter and volume spike.
# Uses daily VWAP for trend direction, Camarilla levels for precise breakout entries,
# and volume surge for confirmation. Designed to work in both bull (breakouts above R1)
# and bear (breakdowns below S1). Target: 20-40 trades/year to avoid fee drag.
name = "4h_Camarilla_R1_S1_Breakout_1dVWAP_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate VWAP for daily timeframe
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    pv = typical_price_1d * df_1d['volume'].values
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(df_1d['volume'].values)
    vwap = np.divide(cum_pv, cum_vol, out=np.zeros_like(cum_pv), where=cum_vol!=0)
    
    # Calculate Camarilla levels for 4h timeframe (based on previous day's OHLC)
    # We need daily OHLC to calculate Camarilla levels for intraday periods
    # Since we're on 4h timeframe, we'll use the previous day's OHLC for all 4h bars of current day
    # Extract daily OHLC from 1d data
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r1 = np.zeros_like(daily_close)
    camarilla_s1 = np.zeros_like(daily_close)
    camarilla_r2 = np.zeros_like(daily_close)
    camarilla_s2 = np.zeros_like(daily_close)
    
    for i in range(len(daily_close)):
        # Camarilla formulas
        range_val = daily_high[i] - daily_low[i]
        camarilla_r1[i] = daily_close[i] + range_val * 1.1 / 12
        camarilla_s1[i] = daily_close[i] - range_val * 1.1 / 12
        camarilla_r2[i] = daily_close[i] + range_val * 1.1 / 6
        camarilla_s2[i] = daily_close[i] - range_val * 1.1 / 6
    
    # Align daily Camarilla levels to 4h timeframe
    # Each 4h bar gets the Camarilla levels from the same day
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    
    # VWAP trend: price above/below VWAP indicates trend
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    
    # Volume confirmation: volume > 1.8x 20-period EMA (moderate threshold)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.8 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 0  # Can start from first bar as Camarilla levels are available
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vwap_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above R1 + price > VWAP (uptrend) + volume spike
            if (price > r1_aligned[i] and price > vwap_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 + price < VWAP (downtrend) + volume spike
            elif (price < s1_aligned[i] and price < vwap_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below R1 or VWAP turns down
            if price < r1_aligned[i] or price < vwap_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above S1 or VWAP turns up
            if price > s1_aligned[i] or price > vwap_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals