#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout + 1d EMA34 trend + volume confirmation
- Long: Close breaks above Camarilla R3 + price > 1d EMA34 (uptrend) + volume > 1.5x 20-period average
- Short: Close breaks below Camarilla S3 + price < 1d EMA34 (downtrend) + volume > 1.5x 20-period average
- Exit: Close retouches Camarilla H3/L3 (midpoint) OR trend reversal
- Uses discrete position sizing (0.25) to minimize fee churn
- Target: 12-37 trades/year (50-150 over 4 years) to avoid fee drag
- Camarilla levels provide precise intraday structure; breakouts with volume and trend filter work in both bull and bear markets
- Using 1d EMA34 as HTF trend filter for better alignment with 12h timeframe
"""

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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical price = (H+L+C)/3
    # Range = H - L
    # R3 = C + (H-L)*1.1/2
    # S3 = C - (H-L)*1.1/2
    # H3 = C + (H-L)*1.1/4
    # L3 = C - (H-L)*1.1/4
    
    # Need previous bar's OHLC for today's Camarilla levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # Set first value to NaN since no previous bar
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    typical_price = (prev_high + prev_low + prev_close) / 3.0
    price_range = prev_high - prev_low
    
    camarilla_r3 = typical_price + price_range * 1.1 / 2.0
    camarilla_s3 = typical_price - price_range * 1.1 / 2.0
    camarilla_h3 = typical_price + price_range * 1.1 / 4.0
    camarilla_l3 = typical_price - price_range * 1.1 / 4.0
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA34
        uptrend = close[i] > ema34_aligned[i]
        downtrend = close[i] < ema34_aligned[i]
        
        # Camarilla breakout signals with trend filter and volume confirmation
        # Long: Close breaks above R3 + uptrend + volume spike
        # Short: Close breaks below S3 + downtrend + volume spike
        long_signal = (close[i] > camarilla_r3[i] and 
                      uptrend and
                      volume[i] > 1.5 * vol_ma[i])
        
        short_signal = (close[i] < camarilla_s3[i] and 
                       downtrend and
                       volume[i] > 1.5 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Close retouches H3/L3 (midpoint) OR trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Close retouches H3 or trend turns down
                if (close[i] <= camarilla_h3[i] or 
                    not uptrend):
                    exit_signal = True
            elif position == -1:
                # Exit short: Close retouches L3 or trend turns up
                if (close[i] >= camarilla_l3[i] or 
                    not downtrend):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0