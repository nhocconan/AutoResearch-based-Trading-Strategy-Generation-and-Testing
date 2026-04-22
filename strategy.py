#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1-hour 3-bar pullback strategy with 4-hour trend filter (4h close > 20-period EMA) 
    # and volume confirmation. Takes long on pullbacks in uptrend, short on bounces in downtrend.
    # Works in bull/bear via trend filter: only trades in direction of 4h trend.
    # Targets ~20-30 trades/year to minimize fee drag while capturing meaningful moves.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend filter (20-period EMA)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Volume spike filter (20-period on 1h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(30, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 3-bar pullback in uptrend (close < low of previous 2 bars) with volume + 4h uptrend
            if (i >= 2 and 
                close[i] < low[i-1] and 
                close[i] < low[i-2] and 
                vol_spike[i] and 
                close[i] > ema20_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: 3-bar bounce in downtrend (close > high of previous 2 bars) with volume + 4h downtrend
            elif (i >= 2 and 
                  close[i] > high[i-1] and 
                  close[i] > high[i-2] and 
                  vol_spike[i] and 
                  close[i] < ema20_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: Trend reversal (price crosses 4h EMA20) or opposite 3-bar setup
            if position == 1:
                if close[i] < ema20_4h_aligned[i] or \
                   (i >= 2 and close[i] > high[i-1] and close[i] > high[i-2]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if close[i] > ema20_4h_aligned[i] or \
                   (i >= 2 and close[i] < low[i-1] and close[i] < low[i-2]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_3Bar_Pullback_4hEMA20_Trend_Volume_Session_v1"
timeframe = "1h"
leverage = 1.0