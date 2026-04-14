#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams %R (14) for mean reversion in ranging markets.
# Long when daily Williams %R crosses above -80 (oversold) AND 12h RSI(14) < 50 AND price > 12h EMA(50).
# Short when daily Williams %R crosses below -20 (overbought) AND 12h RSI(14) > 50 AND price < 12h EMA(50).
# Exit when Williams %R returns to -50 (mean) OR 12h RSI crosses 50 in opposite direction.
# Williams %R identifies overextended moves in daily timeframe, 12h EMA filters trend direction,
# 12h RSI confirms momentum alignment. Designed for ranging markets with clear mean reversion.
# Target: 12-37 trades/year per symbol (48-148 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load daily data ONCE for Williams %R
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate Williams %R (14-period)
    lookback = 14
    highest_high = np.full(len(high_daily), np.nan)
    lowest_low = np.full(len(low_daily), np.nan)
    
    for i in range(lookback - 1, len(high_daily)):
        highest_high[i] = np.max(high_daily[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low_daily[i - lookback + 1:i + 1])
    
    williams_r = np.full(len(close_daily), np.nan)
    for i in range(lookback - 1, len(close_daily)):
        if highest_high[i] - lowest_low[i] != 0:
            williams_r[i] = (highest_high[i] - close_daily[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Load 12h data ONCE for EMA and RSI
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50)
    ema_50 = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema_50[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema_50[i] = (close_12h[i] * 2 + ema_50[i-1] * 49) / 51
    
    # Calculate 12h RSI(14)
    delta = np.diff(close_12h)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_12h), np.nan)
    avg_loss = np.full(len(close_12h), np.nan)
    
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full(len(close_12h), np.nan)
    rsi = np.full(len(close_12h), np.nan)
    for i in range(13, len(avg_gain)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100 if avg_gain[i] > 0 else 0
    
    # Align indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_daily, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Need daily and 12h data
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Previous Williams %R for crossover detection
        prev_williams = williams_r_aligned[i-1] if i > 0 else -50
        
        if position == 0:
            # Look for mean reversion entries
            # Long: Williams %R crosses above -80 (from below) AND RSI < 50 AND price > EMA50
            if (prev_williams <= -80 and williams_r_aligned[i] > -80 and
                rsi_aligned[i] < 50 and
                close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: Williams %R crosses below -20 (from above) AND RSI > 50 AND price < EMA50
            elif (prev_williams >= -20 and williams_r_aligned[i] < -20 and
                  rsi_aligned[i] > 50 and
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to -50 OR RSI crosses above 50
            if (williams_r_aligned[i] >= -50 or 
                rsi_aligned[i] >= 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to -50 OR RSI crosses below 50
            if (williams_r_aligned[i] <= -50 or 
                rsi_aligned[i] <= 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_WilliamsR_MeanReversion_v1"
timeframe = "12h"
leverage = 1.0