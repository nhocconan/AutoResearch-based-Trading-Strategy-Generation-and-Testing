#!/usr/bin/env python3
"""
6h Trend Reversal with 1d Heikin-Ashi Trend Filter and Volume Confirmation
Hypothesis: In 6h timeframes, reversals occur when price rejects key levels. 
Heikin-Ashi smooths price action to identify true trend direction on 1d.
Enter counter-trend when HA shows exhaustion but price rejects extremes with volume.
Works in bull (sell exhaustion at highs) and bear (buy exhaustion at lows).
Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ha_reversal_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Get 1d data for Heikin-Ashi calculation
    df_1d = get_htf_data(prices, '1d')
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Calculate Heikin-Ashi values for 1d
    ha_close = np.empty_like(c_1d)
    ha_open = np.empty_like(o_1d)
    ha_high = np.empty_like(h_1d)
    ha_low = np.empty_like(l_1d)
    
    ha_close[0] = (o_1d[0] + h_1d[0] + l_1d[0] + c_1d[0]) / 4
    ha_open[0] = o_1d[0]
    ha_high[0] = h_1d[0]
    ha_low[0] = l_1d[0]
    
    for i in range(1, len(c_1d)):
        ha_close[i] = (o_1d[i] + h_1d[i] + l_1d[i] + c_1d[i]) / 4
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
        ha_high[i] = max(h_1d[i], ha_open[i], ha_close[i])
        ha_low[i] = min(l_1d[i], ha_open[i], ha_close[i])
    
    # HA trend: 1 for bullish (close > open), -1 for bearish (close < open)
    ha_trend = np.where(ha_close > ha_open, 1, -1)
    
    # Align HA trend to 6h timeframe
    ha_trend_aligned = align_htf_to_ltf(prices, df_1d, ha_trend)
    
    # ATR-based reversal zones: 1.5 * ATR from recent highs/lows
    lookback = 10
    hh = np.full(n, np.nan)  # highest high
    ll = np.full(n, np.nan)  # lowest low
    
    for i in range(lookback, n):
        hh[i] = np.max(h[i-lookback:i])
        ll[i] = np.min(l[i-lookback:i])
    
    # Dynamic resistance/support levels
    resistance = hh - 1.5 * atr
    support = ll + 1.5 * atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0
    
    # Start from warmup period
    start = max(20, lookback + 5)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(ha_trend_aligned[i]) or np.isnan(resistance[i]) or np.isnan(support[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Volume filter: current volume > 1.3x average over last 20 periods
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > vol_ma * 1.3
        else:
            volume_filter = False
        
        # Session filter: 00-23 UTC (all hours for 6h)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        session_filter = True  # 6h bars cover all sessions
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches resistance OR HA turns bearish
            # Stoploss: price drops 2*ATR below entry
            if (close[i] >= resistance[i] or
                ha_trend_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.25
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: price reaches support OR HA turns bullish
            # Stoploss: price rises 2*ATR above entry
            if (close[i] <= support[i] or
                ha_trend_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.25
            bars_since_exit += 1
        else:
            # Look for reversal entries
            # Minimum holding period: only allow new entry after 4 bars flat
            if bars_since_exit >= 4:
                # Bullish reversal: price rejects support with bullish HA exhaustion
                # Price touches/slightly breaks support but HA shows bearish exhaustion (potential bullish reversal)
                touches_support = low[i] <= support[i] * 1.002  # Allow small penetration
                ha_bearish = ha_trend_aligned[i] == -1
                
                # Bearish reversal: price rejects resistance with bearish HA exhaustion
                # Price touches/slightly breaks resistance but HA shows bullish exhaustion (potential bearish reversal)
                touches_resistance = high[i] >= resistance[i] * 0.998  # Allow small penetration
                ha_bullish = ha_trend_aligned[i] == 1
                
                # Long: bullish reversal at support with volume
                if touches_support and ha_bearish and volume_filter and session_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                # Short: bearish reversal at resistance with volume
                elif touches_resistance and ha_bullish and volume_filter and session_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_exit = 0
                else:
                    signals[i] = 0.0
                    bars_since_exit += 1
            else:
                signals[i] = 0.0
                bars_since_exit += 1
    
    return signals