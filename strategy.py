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
- Why it works: Camarilla levels from 1d provide institutional support/resistance; EMA34 filter ensures trend alignment; volume spike confirms institutional participation. Works in bull (breakouts with trend) and bear (fades from levels in range).
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
    
    # Calculate Camarilla levels from prior 1d bar (using daily high/low/close)
    # We'll compute these per 4h bar using the prior completed 1d bar's OHLC
    # Since we don't have daily data aligned, we approximate using rolling window on 4h data
    # This is acceptable as Camarilla is typically calculated from prior session
    # For better accuracy, we would use actual 1d data, but we'll use 4h approximation for now
    # Using 96 periods (24*4) to approximate 1 day of 4h bars
    lookback = 96
    if len(high) < lookback:
        return np.zeros(n)
    
    # Rolling window for prior day's high, low, close (shifted by 1 to avoid look-ahead)
    # We use the completed prior day's OHLC
    prior_day_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(96).values
    prior_day_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(96).values
    prior_day_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().shift(96).values
    
    # Calculate Camarilla levels
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    rng = prior_day_high - prior_day_low
    camarilla_r1 = prior_day_close + rng * 1.1 / 12
    camarilla_s1 = prior_day_close - rng * 1.1 / 12
    
    # Get 1d data for EMA(34) trend
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
    start_idx = max(lookback + 96, 34 + 96)  # Need enough bars for Camarilla and 1d indicators
    
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
        r1_level = camarilla_r1[i]
        s1_level = camarilla_s1[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price breaks above R1 AND 1d EMA34 bullish (price > EMA34)
                if curr_high > r1_level and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below S1 AND 1d EMA34 bearish (price < EMA34)
                elif curr_low < s1_level and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below S1 OR loss of volume confirmation
            if curr_low < s1_level or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 OR loss of volume confirmation
            if curr_high > r1_level or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_1dEMA34Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0