#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation
# Long when price breaks above Donchian upper band AND 1d EMA50 > EMA200 AND volume > 1.5x avg
# Short when price breaks below Donchian lower band AND 1d EMA50 < EMA200 AND volume > 1.5x avg
# Exit when price crosses 10-period EMA on 4h or ATR-based stoploss
# Targets 80-150 total trades over 4 years (20-38/year) with focus on strong trending moves
# Works in bull markets via long breakouts and bear markets via short breakdowns

name = "4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_up = highest_high.values
    donchian_low = lowest_low.values
    
    # 10-period EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # 1d EMA50 and EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(daily_close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align daily EMAs to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(donchian_up[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_10[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or np.isnan(volume_threshold[i]) or \
           np.isnan(atr[i]):
            if position != 0:
                # Maintain position until exit signal
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Update trailing stop for existing positions
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry price OR price crosses below EMA10
            if close[i] < entry_price - 2.0 * atr[i] or close[i] < ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry price OR price crosses above EMA10
            if close[i] > entry_price + 2.0 * atr[i] or close[i] > ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for new entries
            # Long: price breaks above Donchian upper band + uptrend (EMA50 > EMA200) + volume
            if (close[i] > donchian_up[i] and 
                ema_50_aligned[i] > ema_200_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian lower band + downtrend (EMA50 < EMA200) + volume
            elif (close[i] < donchian_low[i] and 
                  ema_50_aligned[i] < ema_200_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals