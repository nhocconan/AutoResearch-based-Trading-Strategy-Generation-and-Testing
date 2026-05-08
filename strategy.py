#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily VWAP reversion with weekly trend filter and volume confirmation
# Long when price closes below VWAP, weekly trend up, and volume above average
# Short when price closes above VWAP, weekly trend down, and volume above average
# VWAP acts as dynamic support/resistance; mean reversion in ranging markets
# Weekly trend filter ensures alignment with higher timeframe momentum
# Volume confirmation avoids low-liquidity false signals
# Targets 30-80 total trades over 4 years (7-20/year) for low frequency and high edge

name = "1d_VWAP_Reversion_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    weekly_close = df_1w['close'].values
    ema20_1w = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate VWAP: cumulative (price * volume) / cumulative volume
    # Using typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    # Avoid division by zero
    vwap = np.divide(cum_pv, cum_vol, out=np.zeros_like(cum_pv), where=cum_vol!=0)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for VWAP and EMA stability
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema20_1w_val = ema20_1w_aligned[i]
        price = close[i]
        vwap_val = vwap[i]
        vol_filt = volume_filter[i]
        
        if position == 0:
            # Enter long: price closes below VWAP, weekly uptrend, volume confirmation
            if price < vwap_val and price > ema20_1w_val and vol_filt:
                signals[i] = 0.25
                position = 1
            # Enter short: price closes above VWAP, weekly downtrend, volume confirmation
            elif price > vwap_val and price < ema20_1w_val and vol_filt:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back above VWAP or weekly trend turns down
            if price > vwap_val or price < ema20_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back below VWAP or weekly trend turns up
            if price < vwap_val or price > ema20_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals