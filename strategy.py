#!/usr/bin/env python3
# Hypothesis: 4h Donchian channel breakout (20-period) with 1d EMA50 trend filter and volume confirmation (>1.8x 20-bar avg volume).
# Uses Donchian breakouts for momentum capture, 1d EMA50 for trend alignment (works in bull/bear via trend filter),
# and volume spike to avoid false breakouts. Designed for low trade frequency (target: 20-50 total 4h trades/year)
# to minimize fee drag while capturing strong trend moves. Includes ATR-based stoploss via signal=0 on close < highest - 2.5*ATR.

name = "4h_Donchian20_1dEMA50_Volume_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().shift(1).values
    
    # Calculate ATR (14-period) for dynamic stoploss
    lookback_atr = 14
    tr1 = pd.Series(high).rolling(window=lookback_atr).max() - pd.Series(low).rolling(window=lookback_atr).min()
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=lookback_atr, adjust=False, min_periods=lookback_atr).mean().values
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_dc, lookback_vol, lookback_atr) + 1, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper band, close > 1d EMA50, volume spike (>1.8x avg)
            if (high[i] > highest_high[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25  # Position size: 25% of capital
                position = 1
            # SHORT: Price breaks below Donchian lower band, close < 1d EMA50, volume spike (>1.8x avg)
            elif (low[i] < lowest_low[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25  # Position size: 25% of capital
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close if price drops below highest_high - 2.5*ATR (trailing stop)
            # or if trend reverses (close < 1d EMA50)
            trailing_stop = highest_high[i] - 2.5 * atr[i]
            if close[i] < trailing_stop or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # EXIT SHORT: Close if price rises above lowest_low + 2.5*ATR (trailing stop)
            # or if trend reverses (close > 1d EMA50)
            trailing_stop = lowest_low[i] + 2.5 * atr[i]
            if close[i] > trailing_stop or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals