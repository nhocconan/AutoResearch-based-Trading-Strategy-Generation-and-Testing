#!/usr/bin/env python3
"""
6h Bollinger Band Squeeze Breakout with Volume Confirmation and Daily Trend Filter
Hypothesis: Bollinger Band squeeze (low volatility) followed by breakout with volume 
confirmation and alignment with daily trend (price above/below daily EMA50) captures 
explosive moves in both bull and bear markets. Works in ranging markets (squeeze) 
and trending markets (breakout). Target: 15-25 trades/year to minimize fee drag.
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
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma20 + bb_std * std20
    lower_bb = sma20 - bb_std * std20
    bb_width = (upper_bb - lower_bb) / sma20  # Normalized width
    
    # Bollinger Band Squeeze: width below 20-period average width
    avg_width = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < 0.5 * avg_width  # Squeeze when width is less than 50% of average
    
    # Volume confirmation: volume > 2x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > 2 * vol_ema
    
    # Daily trend filter: EMA50 from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators (max of 20,20,20,50)
    
    for i in range(start_idx, n):
        if (np.isnan(sma20[i]) or np.isnan(std20[i]) or np.isnan(avg_width[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_bb[i]
        lower = lower_bb[i]
        is_squeeze = squeeze[i]
        vol_ok = vol_confirm[i]
        daily_ema = ema50_1d_aligned[i]
        
        if position == 0:
            # Look for breakout after squeeze with volume confirmation
            if is_squeeze and vol_ok:
                # Bullish breakout: price above upper BB and above daily EMA50
                if price > upper and price > daily_ema:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price below lower BB and below daily EMA50
                elif price < lower and price < daily_ema:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit if price returns to middle Bollinger Band or squeeze breaks down
            if price < sma20[i] or (not is_squeeze and price < upper):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to middle Bollinger Band or squeeze breaks down
            if price > sma20[i] or (not is_squeeze and price > lower):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Bollinger_Squeeze_Breakout_Volume_DailyTrend"
timeframe = "6h"
leverage = 1.0