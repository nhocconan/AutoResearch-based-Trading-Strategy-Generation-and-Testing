#!/usr/bin/env python3
"""
12h_engulfing_1d_trend_volume_v1
Hypothesis: On 12h timeframe, use bullish/bearish engulfing candles for entry signals, filtered by 1d EMA trend (50/200) and volume confirmation (>1.5x 20-period average). Engulfing patterns provide high-probability reversals, while 1d trend filter ensures alignment with higher-timeframe momentum. Volume confirms institutional participation. Designed for low trade frequency (12-37/year) to minimize fee drag in ranging/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_engulfing_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for trend filter (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 and EMA200 on daily close
    daily_close = df_1d['close'].values
    daily_close_s = pd.Series(daily_close)
    ema50_1d = daily_close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_1d = daily_close_s.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align to 12h timeframe (shifted by 1 day to avoid look-ahead)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA200 warmup
        # Skip if required data not available
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter from 1d: up if EMA50 > EMA200, down if EMA50 < EMA200
        trend_up = ema50_1d_aligned[i] > ema200_1d_aligned[i]
        trend_down = ema50_1d_aligned[i] < ema200_1d_aligned[i]
        
        # Bullish engulfing: current green candle fully engulfs previous red candle
        bullish_engulf = (close[i] > open_price[i]) and \
                         (open_price[i-1] > close[i-1]) and \
                         (close[i] >= open_price[i-1]) and \
                         (open_price[i] <= close[i-1])
        
        # Bearish engulfing: current red candle fully engulfs previous green candle
        bearish_engulf = (close[i] < open_price[i]) and \
                         (open_price[i-1] < close[i-1]) and \
                         (open_price[i] >= close[i-1]) and \
                         (close[i] <= open_price[i-1])
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit on trend reversal (EMA50 < EMA200)
            if ema50_1d_aligned[i] < ema200_1d_aligned[i]:
                exit_long = True
            # Exit when volume drops below average
            elif volume[i] < vol_ma[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit on trend reversal (EMA50 > EMA200)
            if ema50_1d_aligned[i] > ema200_1d_aligned[i]:
                exit_short = True
            # Exit when volume drops below average
            elif volume[i] < vol_ma[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: bullish engulfing, 1d trend up, volume confirmation
            long_entry = bullish_engulf and trend_up and vol_confirm
            
            # Short entry: bearish engulfing, 1d trend down, volume confirmation
            short_entry = bearish_engulf and trend_down and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals