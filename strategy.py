#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA34 trend filter and volume confirmation
    # Camarilla levels provide mathematically derived support/resistance. Breakouts with volume
    # confirm institutional participation. 12h EMA34 ensures alignment with higher timeframe trend.
    # Session filter (08-20 UTC) avoids low-liquidity periods. Target 20-50 trades/year.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla Pivot Points (based on previous day's range)
    # Using 4h data to calculate daily pivot: previous day's high/low/close
    # We'll use 16 periods back (4h * 6 = 1 day) for previous day's values
    prev_high = np.roll(high_4h, 16)  # Previous day's high
    prev_low = np.roll(low_4h, 16)    # Previous day's low
    prev_close = np.roll(close_4h, 16) # Previous day's close
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Load 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if data not ready or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above R1 with volume + price above 12h EMA34 (uptrend)
            if close[i] > r1_aligned[i] and vol_spike[i] and close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S1 with volume + price below 12h EMA34 (downtrend)
            elif close[i] < s1_aligned[i] and vol_spike[i] and close[i] < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Camarilla level or trend reversal vs 12h EMA34
            if position == 1:
                if close[i] < s1_aligned[i] or close[i] < ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > r1_aligned[i] or close[i] > ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA34_Volume_Session_v1"
timeframe = "4h"
leverage = 1.0