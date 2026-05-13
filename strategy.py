#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume spike (>2.0x 20-bar avg), and choppiness regime (CHOP > 61.8 = range → mean reversion, CHOP < 38.2 = trend → trend follow). Uses discrete position sizing (0.0, ±0.30) to minimize fee churn. Designed for BTC/ETH robustness: Camarilla breakouts capture institutional order flow, EMA34 filter avoids counter-trend trades, volume spike confirms participation, and chop regime adapts to market state. Targets 20-50 trades/year on 4h timeframe.

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeChopRegime_v3"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Camarilla levels (R3/S3) from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate volume spike confirmation (20-period avg)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Calculate Choppiness Index regime filter (14-period) on 4h data
    atr = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr.sum() / (highest_high - lowest_low)) / np.log10(14)
    # Handle division by zero and edge cases
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)
    chop = np.where(np.isnan(chop), 50, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(avg_volume[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine regime: CHOP > 61.8 = range (mean reversion), CHOP < 38.2 = trend (trend follow)
            in_range = chop[i] > 61.8
            in_trend = chop[i] < 38.2
            
            # LONG conditions
            long_breakout = close[i] > camarilla_r3_aligned[i]
            long_pullback = close[i] < camarilla_r3_aligned[i] and close[i] > camarilla_s3_aligned[i]
            long_volume = volume[i] > 2.0 * avg_volume[i]
            long_trend_filter = close[i] > ema_34_1d_aligned[i]
            
            # SHORT conditions
            short_breakout = close[i] < camarilla_s3_aligned[i]
            short_pullback = close[i] > camarilla_s3_aligned[i] and close[i] < camarilla_r3_aligned[i]
            short_volume = volume[i] > 2.0 * avg_volume[i]
            short_trend_filter = close[i] < ema_34_1d_aligned[i]
            
            if in_range:
                # Range regime: mean reversion at Camarilla extremes
                if long_pullback and long_volume:
                    signals[i] = 0.30
                    position = 1
                elif short_pullback and short_volume:
                    signals[i] = -0.30
                    position = -1
                else:
                    signals[i] = 0.0
            elif in_trend:
                # Trend regime: breakout in direction of trend
                if long_breakout and long_volume and long_trend_filter:
                    signals[i] = 0.30
                    position = 1
                elif short_breakout and short_volume and short_trend_filter:
                    signals[i] = -0.30
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Choppy regime (38.2 <= CHOP <= 61.8): no trading
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches Camarilla S3 (mean reversion target) OR breaks below R3 (failure)
            if close[i] <= camarilla_s3_aligned[i] or close[i] < camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price reaches Camarilla R3 (mean reversion target) OR breaks above S3 (failure)
            if close[i] >= camarilla_r3_aligned[i] or close[i] > camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals