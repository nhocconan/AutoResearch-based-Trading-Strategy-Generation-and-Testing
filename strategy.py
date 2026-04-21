#!/usr/bin/env python3
"""
Hypothesis: 1h session-based momentum with 4h trend and volume confirmation.
In strong trends (4h EMA20), price pulls back to 1h EMA21 during 8-20 UTC session.
Long on bounce from EMA21 with volume confirmation; short on rejection.
Uses 4h for direction, 1h for timing, session filter to reduce noise.
Targets 15-30 trades/year by requiring trend alignment + session + volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Session filter: 8-20 UTC (precomputed)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA20 for trend direction
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1h EMA21 for pullback entry
    close = prices['close'].values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if not in session or indicators not ready
        if not in_session[i] or np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_21[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        ema_trend = ema_20_4h_aligned[i]
        ema_price = ema_21[i]
        vol = vol_ratio[i]
        
        if position == 0:
            # Enter long: uptrend + pullback to EMA21 + volume
            if (price_close > ema_trend and  # uptrend
                abs(price_close - ema_price) / ema_price < 0.005 and  # near EMA21 (0.5%)
                vol > 1.5):
                signals[i] = 0.20
                position = 1
            # Enter short: downtrend + rejection of EMA21 + volume
            elif (price_close < ema_trend and  # downtrend
                  abs(price_close - ema_price) / ema_price < 0.005 and  # near EMA21
                  vol > 1.5):
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: trend reversal or excessive move from EMA21
            if position == 1:
                if (price_close < ema_trend or  # trend broken
                    price_close > ema_price * 1.015):  # 1.5% adverse move
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if (price_close > ema_trend or  # trend broken
                    price_close < ema_price * 0.985):  # 1.5% adverse move
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Session_Pullback_EMA21_4hEMA20_Volume"
timeframe = "1h"
leverage = 1.0