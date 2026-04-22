#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian breakout with 1w EMA34 trend filter and volume confirmation
    # Donchian breakouts capture breakout moves in trending markets. 1w EMA34 ensures alignment with long-term trend.
    # Volume confirmation filters false breakouts. Session filter avoids low-liquidity periods.
    # Target: 12-37 trades/year for 12h timeframe.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Donchian Channels (20-period)
    donchian_period = 20
    upper_donchian = pd.Series(high_12h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_donchian = pd.Series(low_12h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Align Donchian Channels to 12h timeframe (already aligned but keeping for clarity)
    upper_donchian_aligned = align_htf_to_ltf(prices, df_12h, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_12h, lower_donchian)
    
    # Load 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):  # Start after EMA warmup
        # Skip if data not ready or outside session
        if (np.isnan(upper_donchian_aligned[i]) or np.isnan(lower_donchian_aligned[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above upper Donchian with volume + price above 1w EMA34 (uptrend)
            if close[i] > upper_donchian_aligned[i] and vol_spike[i] and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below lower Donchian with volume + price below 1w EMA34 (downtrend)
            elif close[i] < lower_donchian_aligned[i] and vol_spike[i] and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level or trend reversal vs 1w EMA34
            if position == 1:
                if close[i] < lower_donchian_aligned[i] or close[i] < ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_donchian_aligned[i] or close[i] > ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1wEMA34_Volume_Session_v1"
timeframe = "12h"
leverage = 1.0