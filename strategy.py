#!/usr/bin/env python3
"""
6h_WeeklyPivot_PriceAction_Verification
Hypothesis: Weekly pivot levels (calculated from prior week's OHLC) act as strong support/resistance.
In trending markets, price pulls back to weekly pivot levels (R1/S1, R2/S2) before continuing.
In ranging markets, price respects weekly pivot levels as boundaries.
Entry: Limit orders at weekly pivot levels with rejection candlestick patterns (pin bar, engulfing).
Exit: Opposite pivot level or trend reversal.
Uses volume confirmation to avoid false breakouts.
Designed for 6H timeframe to reduce noise and capture multi-day swings.
Target: 15-35 trades/year per symbol.
"""

name = "6h_WeeklyPivot_PriceAction_Verification"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly pivot points from prior week's OHLC
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot: PP = (H + L + C) / 3
    # R1 = 2*PP - L, S1 = 2*PP - H
    # R2 = PP + (H - L), S2 = PP - (H - L)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6H timeframe (already delayed by weekly close)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Trend filter: 24-period EMA (~4 days on 6H)
    ema_24 = pd.Series(close).ewm(span=24, adjust=False, min_periods=24).mean().values
    uptrend = close > ema_24
    downtrend = close < ema_24
    
    # Volume confirmation: volume > 1.5 * 24-period average
    vol_ma = np.zeros(n)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    volume_conf = volume > 1.5 * vol_ma
    
    # Candlestick patterns for rejection/continuation
    # Bullish engulfing: current green candle fully engulfs previous red candle
    bullish_engulf = (close > open_) & (open_ < close) & \
                     (close[:-1] < open_[:-1]) & (open_[:-1] > close[:-1]) & \
                     (close >= open_[:-1]) & (open_ <= close[:-1])
    # Shift to align with current candle
    bullish_engulf = np.concatenate([[False], bullish_engulf[:-1]])
    
    # Bearish engulfing: current red candle fully engulfs previous green candle
    bearish_engulf = (close < open_) & (open_ > close) & \
                     (close[:-1] > open_[:-1]) & (open_[:-1] < close[:-1]) & \
                     (close <= open_[:-1]) & (open_ >= close[:-1])
    bearish_engulf = np.concatenate([[False], bearish_engulf[:-1]])
    
    # Pin bar patterns
    # Bullish pin: long lower wick, small body, close near high
    body_size = np.abs(close - open_)
    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low
    bullish_pin = (lower_wick > 2 * body_size) & (upper_wick < 0.5 * body_size)
    # Bearish pin: long upper wick
    bearish_pin = (upper_wick > 2 * body_size) & (lower_wick < 0.5 * body_size)
    
    # Combine rejection signals
    bullish_reject = bullish_engulf | bullish_pin
    bearish_reject = bearish_engulf | bearish_pin
    
    # Align open prices for candle calculations
    open_ = prices['open'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if no weekly data yet
        if np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
            signals[i] = 0.0
            continue
            
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        
        if position == 0:
            # LONG: price near support with bullish rejection
            near_s1 = abs(low[i] - s1_val) / s1_val < 0.005  # within 0.5%
            near_s2 = abs(low[i] - s2_val) / s2_val < 0.005
            
            if ((near_s1 or near_s2) and bullish_reject[i] and uptrend[i] and volume_conf[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price near resistance with bearish rejection
            elif ((abs(high[i] - r1_val) / r1_val < 0.005 or abs(high[i] - r2_val) / r2_val < 0.005) and 
                  bearish_reject[i] and downtrend[i] and volume_conf[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price reaches resistance or trend turns down
            if (abs(high[i] - r1_val) / r1_val < 0.005 or 
                abs(high[i] - r2_val) / r2_val < 0.005 or 
                not uptrend[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price reaches support or trend turns up
            if (abs(low[i] - s1_val) / s1_val < 0.005 or 
                abs(low[i] - s2_val) / s2_val < 0.005 or 
                not downtrend[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals