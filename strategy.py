#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for ATR and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on daily
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 34-period EMA on daily close
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter (20-period on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 4-hour timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(atr_14_aligned[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > EMA34, EMA50_12h > EMA34 (uptrend), and volatility breakout
            if (close[i] > ema_34_aligned[i] and 
                ema_50_12h_aligned[i] > ema_34_aligned[i] and
                close[i] > close[i-1] + 1.5 * atr_14_aligned[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < EMA34, EMA50_12h < EMA34 (downtrend), and volatility breakout
            elif (close[i] < ema_34_aligned[i] and 
                  ema_50_12h_aligned[i] < ema_34_aligned[i] and
                  close[i] < close[i-1] - 1.5 * atr_14_aligned[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Reverse signal or volatility contraction
            if position == 1:
                if (close[i] < ema_34_aligned[i] or
                    close[i] < close[i-1] - 1.0 * atr_14_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > ema_34_aligned[i] or
                    close[i] > close[i-1] + 1.0 * atr_14_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_EMA_Volatility_Breakout_12hTrend"
timeframe = "4h"
leverage = 1.0