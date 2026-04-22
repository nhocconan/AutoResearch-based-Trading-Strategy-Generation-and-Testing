#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h 4h/1d trend alignment with volume confirmation and session filter
    # Uses 4h trend direction via price vs 4h EMA50 and 1d trend via price vs 1d EMA50
    # Enters on 1h pullbacks to EMA20 with volume confirmation during active hours (08-20 UTC)
    # Designed for low trade frequency (15-30/year) to minimize fee drag in ranging markets
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1h EMA20 for entry timing
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if data not ready or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(ema20[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Aligned uptrend (price > 4h EMA50 and price > 1d EMA50) +
            #       pullback to 1h EMA20 with volume spike
            if (close[i] > ema50_4h_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and
                close[i] <= ema20[i] * 1.005 and  # Allow small overshoot
                vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: Aligned downtrend (price < 4h EMA50 and price < 1d EMA50) +
            #        pullback to 1h EMA20 with volume spike
            elif (close[i] < ema50_4h_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and
                  close[i] >= ema20[i] * 0.995 and  # Allow small undershoot
                  vol_spike[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: Trend break (price crosses 4h EMA50 in opposite direction)
            if position == 1:
                if close[i] < ema50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if close[i] > ema50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_TrendAlignment_EMA50_4h1d_EMA20_Pullback_Volume"
timeframe = "1h"
leverage = 1.0