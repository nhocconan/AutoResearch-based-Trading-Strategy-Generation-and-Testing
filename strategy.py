#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA50 trend filter and volume spike confirmation.
# Long when: Williams %R(14) crosses above -80 (oversold bounce) AND close > 1d EMA50 AND volume > 1.8x 24-bar average
# Short when: Williams %R(14) crosses below -20 (overbought rejection) AND close < 1d EMA50 AND volume > 1.8x 24-bar average
# Exit via ATR(24) trailing stop: long exit when price < highest_high_since_entry - 2.0 * ATR
#                      short exit when price > lowest_low_since_entry + 2.0 * ATR
# Uses 6h Williams %R for mean reversion edge in ranging markets, 1d EMA50 for HTF trend alignment, volume spike for confirmation
# Discrete sizing 0.25 balances return and fee drag. Target: 50-150 total trades over 4 years = 12-37/year.

name = "6h_WilliamsR_1dEMA50_VolumeSpike_ATRStop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h Williams %R (14)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Calculate 1d EMA50 (HTF trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 24-bar average volume for spike confirmation
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate ATR(24) for trailing stop
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    atr = pd.Series(tr).rolling(window=24, min_periods=1).mean().values
    
    signals = np.zeros(n)
    
    # Track position state for trailing stop
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(24, n):
        # Williams %R crossover signals
        wr_cross_up = williams_r[i-1] <= -80 and williams_r[i] > -80
        wr_cross_down = williams_r[i-1] >= -20 and williams_r[i] < -20
        
        # Volume confirmation
        volume_spike = volume[i] > 1.8 * vol_ma_24[i]
        
        # Trend filter
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Trailing stop logic
        if position_side == 1:  # Long position
            highest_since_entry = max(highest_since_entry, high[i])
            if close[i] < highest_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0  # Stoploss hit
                position_side = 0
                continue
        elif position_side == -1:  # Short position
            lowest_since_entry = min(lowest_since_entry, low[i])
            if close[i] > lowest_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0  # Stoploss hit
                position_side = 0
                continue
        
        # Entry logic (only when flat)
        if position_side == 0:
            if wr_cross_up and trend_up and volume_spike:
                signals[i] = 0.25  # Long 25%
                position_side = 1
                highest_since_entry = high[i]
            elif wr_cross_down and trend_down and volume_spike:
                signals[i] = -0.25  # Short 25%
                position_side = -1
                lowest_since_entry = low[i]
        # Hold current position
        elif position_side == 1:
            signals[i] = 0.25
        elif position_side == -1:
            signals[i] = -0.25
    
    return signals