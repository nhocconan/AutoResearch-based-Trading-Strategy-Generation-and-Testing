#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams %R with 1d EMA34 trend filter and volume confirmation
    # Williams %R identifies overbought/oversold conditions. Mean reversion from extremes
    # works in ranging markets, while EMA34 filter ensures alignment with daily trend.
    # Volume confirmation reduces false signals. Session filter avoids low-liquidity periods.
    # Target: 12-37 trades/year (50-150 total over 4 years) for 12h timeframe.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams %R and EMA34
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R (14-period)
    williams_period = 14
    highest_high = pd.Series(high_1d).rolling(window=williams_period, min_periods=williams_period).max().values
    lowest_low = pd.Series(low_1d).rolling(window=williams_period, min_periods=williams_period).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    
    # EMA34 trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align indicators to 12h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike filter (20-period on 12h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if data not ready or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or
            np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Oversold (Williams %R < -80) + volume + price above EMA34 (uptrend)
            if williams_r_aligned[i] < -80 and vol_spike[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Overbought (Williams %R > -20) + volume + price below EMA34 (downtrend)
            elif williams_r_aligned[i] > -20 and vol_spike[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to neutral zone (-50) or trend reversal vs EMA34
            if position == 1:
                if williams_r_aligned[i] > -50 or close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if williams_r_aligned[i] < -50 or close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA34_Volume_Session_v1"
timeframe = "12h"
leverage = 1.0