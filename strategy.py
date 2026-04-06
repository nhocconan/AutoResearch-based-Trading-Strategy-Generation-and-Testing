#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band Squeeze Breakout with 1d Trend Filter
# Uses Bollinger Band width to identify low volatility periods (squeeze).
# Breakout occurs when price closes outside Bollinger Bands after squeeze.
# Trend filter uses 1d EMA50 to ensure breakouts align with higher timeframe trend.
# Volume confirmation reduces false breakouts.
# Designed for 12h timeframe to target 50-150 trades over 4 years.

name = "12h_bb_squeeze_1d_ema50_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # EMA50 calculation on 1d
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Bollinger Bands (20, 2) on 12h
    bb_length = 20
    bb_mult = 2.0
    
    # Middle band (SMA)
    bb_middle = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    
    # Standard deviation
    bb_std = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    
    # Upper and lower bands
    bb_upper = bb_middle + (bb_std * bb_mult)
    bb_lower = bb_middle - (bb_std * bb_mult)
    
    # Bollinger Band Width (normalized)
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Squeeze condition: BB width below 20-period average
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(bb_length, n):  # Start after BB warmup
        # Skip if required data not available
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below middle band or trend changes
            elif close[i] < bb_middle[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above middle band or trend changes
            elif close[i] > bb_middle[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Bollinger Band breakout after squeeze
            # Long: price closes above upper band after squeeze, uptrend, volume spike
            if (squeeze[i-1] and  # was in squeeze previous bar
                close[i] > bb_upper[i] and  # breakout above upper band
                close[i] > ema50_1d_aligned[i] and  # above 1d EMA50 (uptrend)
                volume[i] > 1.5 * volume_ma[i]):  # volume confirmation
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price closes below lower band after squeeze, downtrend, volume spike
            elif (squeeze[i-1] and  # was in squeeze previous bar
                  close[i] < bb_lower[i] and  # breakout below lower band
                  close[i] < ema50_1d_aligned[i] and  # below 1d EMA50 (downtrend)
                  volume[i] > 1.5 * volume_ma[i]):  # volume confirmation
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals