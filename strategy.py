#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Camarilla levels: calculated from prior 1h OHLC; long on break above H3, short on break below L3.
- Volume confirmation: current volume > 1.8 * 20-period volume MA to ensure participation.
- Session filter: 08:00-20:00 UTC to avoid low-liquidity hours.
- Exit: opposite Camarilla level touch (L3 for long, H3 for short) or EMA trend reversal.
- Signal size: 0.20 discrete to minimize fee churn and control drawdown.
Designed to work in both bull and bear markets via 1d trend filter and volatility-adjusted breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF EMA34 to 1h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 34, 20)  # Need enough bars for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08:00-20:00 UTC
        hour = prices.index[i].hour
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from prior 1h bar
        if i == 0:
            continue  # skip first bar
        prior_high = high[i-1]
        prior_low = low[i-1]
        prior_close = close[i-1]
        prior_range = prior_high - prior_low
        
        if prior_range <= 0:
            continue  # skip invalid range
        
        camarilla_h3 = prior_close + prior_range * 1.1 / 4
        camarilla_l3 = prior_close - prior_range * 1.1 / 4
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.8x threshold)
            vol_confirmed = curr_volume > 1.8 * vol_ma[i]
            
            # Determine 1d trend: bullish if close > EMA34, bearish if close < EMA34
            htf_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
            htf_close = htf_close_aligned[i]
            
            trend_bullish = htf_close > ema_34_1d_aligned[i]
            trend_bearish = htf_close < ema_34_1d_aligned[i]
            
            # Long: break above H3 AND 1d trend bullish AND volume confirmed
            if curr_close > camarilla_h3 and trend_bullish and vol_confirmed:
                signals[i] = 0.20
                position = 1
            # Short: break below L3 AND 1d trend bearish AND volume confirmed
            elif curr_close < camarilla_l3 and trend_bearish and vol_confirmed:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long when price touches L3 or trend turns bearish
            if curr_close < camarilla_l3 or htf_close < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short when price touches H3 or trend turns bullish
            if curr_close > camarilla_h3 or htf_close > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0