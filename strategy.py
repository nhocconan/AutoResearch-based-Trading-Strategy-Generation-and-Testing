# Hypothesis: 4h Donchian(20) breakout with 1d volume filter and 1w trend filter. 
# Breakouts in high-volume 1d environments capture institutional moves. 
# 1w EMA filter ensures alignment with longer-term trend, reducing whipsaws in sideways/choppy markets.
# Works in bull (breaks up) and bear (breaks down). Discrete sizing 0.25 to limit drawdown.
# Expected trades: ~25-40/year per symbol, avoiding fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data for context
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    close_1w = df_1w['close'].values
    
    # Calculate 1d Donchian channel (20) for breakout signals
    donch_high_20 = np.full(len(df_1d), np.nan)
    donch_low_20 = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        donch_high_20[i] = np.max(high_1d[i-19:i+1])
        donch_low_20[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 1d volume SMA(20) for volume filter
    volume_1d_series = pd.Series(volume_1d)
    vol_sma_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA(20) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 4h ATR(14) for position sizing (not used in signal, but kept for completeness)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-14:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(vol_sma_20_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume (aligned) > 1.2 * 20-period MA
        # Note: We use 1d volume aligned to 4h, approximating current session's volume context
        vol_filter = volume[i] > 1.2 * vol_sma_20_1d_aligned[i]
        
        # Trend filter: price above/below 1w EMA(20)
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > donch_high_20_aligned[i]
        breakout_short = close[i] < donch_low_20_aligned[i]
        
        # Entry conditions: breakout with volume and trend alignment
        long_entry = breakout_long and vol_filter and uptrend
        short_entry = breakout_short and vol_filter and downtrend
        
        # Exit conditions: price crosses back to opposite Donchian band
        long_exit = close[i] < donch_low_20_aligned[i]
        short_exit = close[i] > donch_high_20_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_1w_donchian_vol_trend_filter_v1"
timeframe = "4h"
leverage = 1.0