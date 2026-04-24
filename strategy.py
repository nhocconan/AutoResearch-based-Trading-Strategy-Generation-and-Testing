#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA(34) trend filter and volume spike confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 1d EMA(34) for trend direction (bullish if price > EMA34, bearish if price < EMA34).
- Volume: Current 4h volume > 2.0 * 20-period volume MA to avoid false breakouts.
- Entry: Long when price breaks above Camarilla R1 AND 1d EMA34 trend bullish AND volume spike.
         Short when price breaks below Camarilla S1 AND 1d EMA34 trend bearish AND volume spike.
- Exit: Opposite Camarilla breakout or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Why it works: Camarilla levels are derived from prior day's range and act as intraday support/resistance.
                In trending markets (EMA filter), breaks often continue. Volume confirms legitimacy.
                Works in bull (long bias) and bear (short bias) due to symmetric logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels (R1, S1) from prior day's OHLC
    # Using 4h data but computing levels based on daily pivot logic
    # We'll use rolling window of 6 bars (6*4h = 24h) to approximate prior day
    roll_high = pd.Series(high).rolling(window=6, min_periods=6).max()
    roll_low = pd.Series(low).rolling(window=6, min_periods=6).min()
    roll_close = pd.Series(close).rolling(window=6, min_periods=6).last()
    
    # Prior day's range
    prior_range = roll_high - roll_low
    # Camarilla R1 and S1
    camarilla_r1 = roll_close + (prior_range * 1.1 / 12)
    camarilla_s1 = roll_close - (prior_range * 1.1 / 12)
    
    # Get 1d data for EMA(34) trend and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1d
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 4h volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 6, 20)  # Need enough 1d bars for EMA34, 6-bar roll for camarilla, 20-bar vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        r1 = camarilla_r1[i]
        s1 = camarilla_s1[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price breaks above Camarilla R1 AND 1d EMA34 bullish (price > EMA34)
                if curr_high > r1 and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below Camarilla S1 AND 1d EMA34 bearish (price < EMA34)
                elif curr_low < s1 and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla S1 OR loss of volume confirmation
            if curr_low < s1 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Camarilla R1 OR loss of volume confirmation
            if curr_high > r1 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_1dEMA34Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0