#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Weekly Pivot Breakout with 12h EMA34 Filter and Volume Confirmation
# Uses weekly pivot levels (R1/S1) from previous week as key support/resistance.
# Long when price breaks above R1 with volume > 2x 20-period avg and 12h EMA34 > price (bullish bias).
# Short when price breaks below S1 with volume > 2x 20-period avg and 12h EMA34 < price (bearish bias).
# Exit when price returns to weekly pivot (PP) or reverses with contrary volume spike.
# Weekly pivots provide structure, EMA34 filters trend, volume confirms breakout strength.
# Works in bull (buy breakouts) and bear (sell breakdowns). Target: 12-25 trades/year per symbol.
name = "6h_WeeklyPivot_R1S1_Breakout_Volume_EMA34"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA34 on 12h for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    # We'll approximate weekly data by resampling logic but using actual daily data for pivot calc
    # For simplicity, we'll use daily data to calculate weekly pivots (standard approach)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate weekly high, low, close from daily data (assuming week starts Monday)
    # Simplified: use rolling window of 5 days for weekly OHLC approximation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high = max of last 5 daily highs
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    # Weekly low = min of last 5 daily lows
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    # Weekly close = last daily close
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).apply(lambda x: x[-1]).values
    
    # Calculate pivot points
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    
    # Align weekly pivots to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Wait for volume MA and EMA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema34 = ema34_12h_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        pp = pp_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R1 with volume spike and bullish 12h EMA34
            if (price > r1 and price > close[i-1] and  # breakout above R1
                vol > 2.0 * vol_ma and                # volume spike
                ema34 > price):                       # bullish bias (EMA34 above price)
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with volume spike and bearish 12h EMA34
            elif (price < s1 and price < close[i-1] and  # breakdown below S1
                  vol > 2.0 * vol_ma and                 # volume spike
                  ema34 < price):                        # bearish bias (EMA34 below price)
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to weekly pivot (PP) or breaks below S1 with volume
            if (price < pp or 
                (price < s1 and vol > 2.0 * vol_ma)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly pivot (PP) or breaks above R1 with volume
            if (price > pp or 
                (price > r1 and vol > 2.0 * vol_ma)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals