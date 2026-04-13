#!/usr/bin/env python3
"""
Hypothesis: 1h RSI mean reversion with 4h trend filter and 1d volume confirmation.
- 4h EMA(50) determines trend direction: long only when price > EMA50, short only when price < EMA50
- 1h RSI(14) for mean reversion entries: long when RSI < 30, short when RSI > 70
- 1d volume spike (volume > 1.5x 20-period average) confirms momentum behind the move
- Session filter: only trade 08:00-20:00 UTC to avoid low-liquidity hours
- Fixed position size: 0.20 (20% of capital) to manage drawdown
- Target: 15-30 trades/year to stay under fee drag limits
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Pre-compute session hours (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% of capital
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(rsi[i]) or 
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        trend_up = close[i] > ema_4h_aligned[i]
        trend_down = close[i] < ema_4h_aligned[i]
        vol_confirm = vol_spike_aligned[i] > 0.5
        
        long_entry = rsi_oversold and trend_up and vol_confirm
        short_entry = rsi_overbought and trend_down and vol_confirm
        
        # Exit when RSI returns to neutral zone (40-60)
        exit_long = position == 1 and rsi[i] > 40
        exit_short = position == -1 and rsi[i] < 60
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_rsi_meanrev_4h_trend_1d_vol"
timeframe = "1h"
leverage = 1.0