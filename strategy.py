# 12h_Keltner_Channel_Breakout_Volume
# Hypothesis: 12h Keltner Channel (EMA10 + ATR(10)*2) breakout with volume confirmation and 1d trend filter (EMA50) to filter false breakouts.
# Works in bull: breakouts above upper band in uptrend; works in bear: breakouts below lower band in downtrend.
# Targets 15-25 trades/year by requiring EMA trend alignment, KC breakout, and volume > 2x 20-period average.

#!/usr/bin/env python3
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
    
    # Get 12h data for Keltner Channel (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate EMA10 and ATR(10) for Keltner Channel on 12h
    ema_12h = pd.Series(close_12h).ewm(span=10, adjust=False).values
    tr_12h = np.maximum(high_12h - low_12h, 
                        np.absolute(high_12h - np.roll(close_12h, 1)),
                        np.absolute(low_12h - np.roll(close_12h, 1)))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr_12h = pd.Series(tr_12h).ewm(span=10, adjust=False).mean().values
    
    # Keltner Channel bounds
    upper_12h = ema_12h + (2 * atr_12h)
    lower_12h = ema_12h - (2 * atr_12h)
    
    # Align KC bounds to 12h timeframe (already correct timeframe, but align for safety)
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 2 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA and EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above upper KC, with volume, and 1d EMA50 uptrend (close > EMA50)
            if (close[i] > upper_12h_aligned[i] and vol_confirm[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower KC, with volume, and 1d EMA50 downtrend (close < EMA50)
            elif (close[i] < lower_12h_aligned[i] and vol_confirm[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns below EMA10 (middle of KC) or breaks below lower KC (failed breakout)
            if (close[i] < ema_12h_aligned[i] or 
                close[i] < lower_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above EMA10 or breaks above upper KC (failed breakout)
            if (close[i] > ema_12h_aligned[i] or 
                close[i] > upper_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Keltner_Channel_Breakout_Volume"
timeframe = "12h"
leverage = 1.0