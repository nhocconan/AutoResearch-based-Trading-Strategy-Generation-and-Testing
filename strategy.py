#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams %R extremes with 1w EMA34 trend filter and volume spike confirmation
# Long when Williams %R < -80 (oversold) in 1w uptrend (close > EMA34) with volume spike (>2.0x average).
# Short when Williams %R > -20 (overbought) in 1w downtrend (close < EMA34) with volume spike.
# Designed for low trade frequency (~12-37/year on 12h) to minimize fee drag while capturing mean reversion in trends.
# Works in bull markets via buying dips in uptrends and in bear markets via selling rallies in downtrends.
# Uses 1d HTF for Williams %R calculation and 1w EMA34 for trend alignment.
# Signal value: 0.30 for discrete position sizing to reduce fee churn.

name = "12h_1dWilliamsR_Extreme_1wEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 14 or len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 1d Williams %R to 12h timeframe (wait for 1d bar to close)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Calculate ATR(14) for dynamic stoploss on 12h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 40  # warmup for EMA(34) and Williams %R
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 40-period average
        if i >= 40:
            vol_ma_40 = np.mean(volume[i-40:i])
        elif i > 0:
            vol_ma_40 = np.mean(volume[:i])
        else:
            vol_ma_40 = 0
        volume_spike = volume[i] > (2.0 * vol_ma_40) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_williams_r = williams_r_aligned[i]
        curr_ema = ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: Williams %R < -80 (oversold) with 1w uptrend (close > EMA34)
                if curr_williams_r < -80 and curr_close > curr_ema:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Williams %R > -20 (overbought) with 1w downtrend (close < EMA34)
                elif curr_williams_r > -20 and curr_close < curr_ema:
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry price OR Williams %R > -20 (overbought reversal)
            if curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_williams_r > -20:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 2.5x ATR above entry
            elif curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry price OR Williams %R < -80 (oversold reversal)
            if curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_williams_r < -80:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 2.5x ATR below entry
            elif curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = -0.30
    
    return signals