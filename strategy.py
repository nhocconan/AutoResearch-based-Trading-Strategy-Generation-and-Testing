#!/usr/bin/env python3
# 12h_1w_1d_Triple_Timeframe_Confluence
# Hypothesis: Combines weekly trend (1w EMA), daily momentum (RSI), and 12h price action (breakout of daily ATR-based channels) for high-conviction trades.
# Weekly EMA filter ensures we trade with the higher timeframe trend, reducing whipsaws in sideways markets.
# Daily RSI > 50 for longs and < 50 for shorts adds momentum confirmation.
# 12h entry uses ATR-based channels (similar to Donchian but adaptive to volatility) to capture breakouts with volatility context.
# Volume confirmation ensures breakouts are supported by participation.
# Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag and work in both bull and bear markets.

name = "12h_1w_1d_Triple_Timeframe_Confluence"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data (trend filter) ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data (momentum and channel) ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly EMA(40) for trend filter
    close_1w = df_1w['close'].values
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Daily RSI(14) for momentum
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14 = rsi_14.fillna(50).values  # Fill NaN with 50 (neutral)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Daily ATR(20) for volatility-adjusted channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_20 = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    
    # Daily typical price for center of channel
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    typical_price_aligned = align_htf_to_ltf(prices, df_1d, typical_price_1d)
    
    # Upper and lower channels: typical price ± 1.5 * ATR
    upper_channel = typical_price_aligned + 1.5 * atr_20_aligned
    lower_channel = typical_price_aligned - 1.5 * atr_20_aligned
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_40_1w_aligned[i]) or np.isnan(rsi_14_aligned[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: above upper channel, weekly uptrend, bullish momentum, volume confirmation
            if (close[i] > upper_channel[i] and 
                close[i] > ema_40_1w_aligned[i] and 
                rsi_14_aligned[i] > 50 and 
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: below lower channel, weekly downtrend, bearish momentum, volume confirmation
            elif (close[i] < lower_channel[i] and 
                  close[i] < ema_40_1w_aligned[i] and 
                  rsi_14_aligned[i] < 50 and 
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below lower channel or weekly trend turns down
            if close[i] < lower_channel[i] or close[i] < ema_40_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above upper channel or weekly trend turns up
            if close[i] > upper_channel[i] or close[i] > ema_40_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals