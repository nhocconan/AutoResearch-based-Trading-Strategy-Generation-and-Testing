#!/usr/bin/env python3
"""
Hypothesis: 12h-based strategy using weekly Bollinger Band squeeze (20, 2) combined with 1-week ATR trend filter and volume confirmation. 
Bollinger squeeze identifies low volatility periods preceding breakouts, while 1-week ATR confirms trend direction. 
Volume ensures breakout conviction. Designed for 15-25 trades/year to minimize fee drag.
Works in bull markets (breakout above upper band in uptrend) and bear markets (breakdown below lower band in downtrend).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands and ATR
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Bollinger Bands (20, 2) on weekly close
    bb_width = np.full(len(close_1w), np.nan)
    bb_upper = np.full(len(close_1w), np.nan)
    bb_lower = np.full(len(close_1w), np.nan)
    bb_middle = np.full(len(close_1w), np.nan)
    
    if len(close_1w) >= 20:
        # Calculate SMA and std dev
        for i in range(19, len(close_1w)):
            sma = np.mean(close_1w[i-19:i+1])
            std = np.std(close_1w[i-19:i+1])
            bb_middle[i] = sma
            bb_upper[i] = sma + 2 * std
            bb_lower[i] = sma - 2 * std
            bb_width[i] = (bb_upper[i] - bb_lower[i]) / bb_middle[i] if bb_middle[i] != 0 else np.nan
    
    # Calculate Bollinger Band squeeze: width < 20th percentile of width (low volatility)
    bb_squeeze = np.full(len(close_1w), np.nan)
    if len(bb_width) >= 50:  # Need enough data for percentile
        # Calculate 50-period rolling 20th percentile of BB width
        for i in range(49, len(bb_width)):
            window = bb_width[i-49:i+1]
            valid_widths = window[~np.isnan(window)]
            if len(valid_widths) >= 10:  # Minimum samples for percentile
                bb_squeeze[i] = np.percentile(valid_widths, 20)
    
    # Squeeze condition: current width < 20th percentile width
    is_squeeze = np.full(len(close_1w), False)
    for i in range(len(close_1w)):
        if not (np.isnan(bb_width[i]) or np.isnan(bb_squeeze[i])):
            is_squeeze[i] = bb_width[i] < bb_squeeze[i]
    
    # Calculate 1-week ATR(14) for trend filter
    atr_1w = np.full(len(close_1w), np.nan)
    if len(high_1w) >= 15:  # Need at least 15 periods for ATR(14)
        tr = np.full(len(high_1w), np.nan)
        for i in range(1, len(high_1w)):
            hl = high_1w[i] - low_1w[i]
            hc = np.abs(high_1w[i] - close_1w[i-1])
            lc = np.abs(low_1w[i] - close_1w[i-1])
            tr[i] = max(hl, hc, lc)
        
        # Calculate ATR using Wilder's smoothing
        if len(tr) >= 15:
            atr_1w[14] = np.nanmean(tr[1:15])  # First ATR
            for i in range(15, len(tr)):
                if not np.isnan(tr[i]):
                    atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    # Calculate 1-week EMA(34) for trend direction
    ema_34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[:34])
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = (close_1w[i] * 2/35) + (ema_34_1w[i-1] * 33/35)
    
    # Align weekly indicators to 12h timeframe
    bb_squeeze_12h = align_htf_to_ltf(prices, df_1w, is_squeeze)
    atr_1w_12h = align_htf_to_ltf(prices, df_1w, atr_1w)
    ema_34_1w_12h = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    bb_upper_12h = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_12h = align_htf_to_ltf(prices, df_1w, bb_lower)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # need squeeze, ATR, EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_squeeze_12h[i]) or np.isnan(atr_1w_12h[i]) or 
            np.isnan(ema_34_1w_12h[i]) or np.isnan(bb_upper_12h[i]) or 
            np.isnan(bb_lower_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter: price above/below 1-week EMA34
        trend_up = close[i] > ema_34_1w_12h[i]
        trend_down = close[i] < ema_34_1w_12h[i]
        
        if position == 0:
            # Long entry: Bollinger breakout above upper band with volume and uptrend
            if (close[i] > bb_upper_12h[i] and 
                bb_squeeze_12h[i] and  # Only trade after squeeze
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: Bollinger breakdown below lower band with volume and downtrend
            elif (close[i] < bb_lower_12h[i] and 
                  bb_squeeze_12h[i] and  # Only trade after squeeze
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to middle Bollinger Band or reverse signal
            if close[i] < bb_middle[i] if not np.isnan(bb_middle[i]) else False:  # Simplified: exit when price < EMA34
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle Bollinger Band or reverse signal
            if close[i] > bb_middle[i] if not np.isnan(bb_middle[i]) else False:  # Simplified: exit when price > EMA34
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_BollingerSqueeze_1wATR34_EMA_Volume"
timeframe = "12h"
leverage = 1.0