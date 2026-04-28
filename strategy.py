#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
# Uses 6h primary timeframe targeting 12-37 trades/year (50-150 total over 4 years).
# Ichimoku components (Tenkan, Kijun, Senkou Span A/B) from 6d data provide institutional support/resistance.
# 1d EMA50 provides primary trend filter: bull when price > EMA50, bear when price < EMA50.
# Volume spike (>2.0x 24-bar average) confirms breakout strength.
# Position size 0.25 for balance between return and drawdown control.
# Discrete levels (0.0, ±0.25) minimize fee churn.
# Ichimoku works in both bull/bear markets: cloud acts as dynamic S/R, TK cross signals momentum shifts.

name = "6h_Ichimoku_CloudBreakout_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to reduce noise
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 6d data for Ichimoku and 1d data for EMA50 trend
    df_6d = get_htf_data(prices, '6d')
    df_1d = get_htf_data(prices, '1d')
    if len(df_6d) < 52 or len(df_1d) < 50:
        return np.zeros(n)
    
    high_6d = df_6d['high'].values
    low_6d = df_6d['low'].values
    close_6d = df_6d['close'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_6d = (pd.Series(high_6d).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low_6d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_6d = (pd.Series(high_6d).rolling(window=26, min_periods=26).max() + 
                pd.Series(low_6d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_a_6d = (tenkan_6d + kijun_6d) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_b_6d = (pd.Series(high_6d).rolling(window=52, min_periods=52).max() + 
                   pd.Series(low_6d).rolling(window=52, min_periods=52).min()) / 2
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    tenkan_6d_aligned = align_htf_to_ltf(prices, df_6d, tenkan_6d.values)
    kijun_6d_aligned = align_htf_to_ltf(prices, df_6d, kijun_6d.values)
    senkou_a_6d_aligned = align_htf_to_ltf(prices, df_6d, senkou_a_6d.values)
    senkou_b_6d_aligned = align_htf_to_ltf(prices, df_6d, senkou_b_6d.values)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h volume spike: >2.0x 24-bar average volume
    volume_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Ensure sufficient history for Ichimoku (52 periods)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_6d_aligned[i]) or
            np.isnan(kijun_6d_aligned[i]) or
            np.isnan(senkou_a_6d_aligned[i]) or
            np.isnan(senkou_b_6d_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA50 direction (price above/below EMA50)
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Ichimoku conditions
        # Bullish: price above cloud AND Tenkan > Kijun (TK cross up)
        bullish_cloud = close[i] > max(senkou_a_6d_aligned[i], senkou_b_6d_aligned[i])
        bullish_tk = tenkan_6d_aligned[i] > kijun_6d_aligned[i]
        bullish_ichimoku = bullish_cloud and bullish_tk
        
        # Bearish: price below cloud AND Tenkan < Kijun (TK cross down)
        bearish_cloud = close[i] < min(senkou_a_6d_aligned[i], senkou_b_6d_aligned[i])
        bearish_tk = tenkan_6d_aligned[i] < kijun_6d_aligned[i]
        bearish_ichimoku = bearish_cloud and bearish_tk
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        long_entry = bullish_ichimoku and price_above_ema and vol_confirm
        short_entry = bearish_ichimoku and price_below_ema and vol_confirm
        
        # Exit conditions: opposite Ichimoku signal (cloud breach + TK cross reversal)
        long_exit = bearish_ichimoku  # Exit long when bearish Ichimoku signal appears
        short_exit = bullish_ichimoku  # Exit short when bullish Ichimoku signal appears
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals