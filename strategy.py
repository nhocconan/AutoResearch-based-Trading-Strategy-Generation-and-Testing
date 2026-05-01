#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and ATR-based stoploss.
# Uses weekly EMA34 for trend filter (bullish when price > EMA34, bearish when price < EMA34).
# Breakout above 20-day high for long, below 20-day low for short, only in trend direction.
# Includes ATR(14) stoploss: exit when price moves 2.5*ATR against position.
# Discrete position sizing (0.30) to balance risk and reward. Target: 30-100 total trades over 4 years.

name = "1d_Donchian20_Breakout_1wEMA34_ATRStop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup for EMA34, Donchian, and ATR
    start_idx = max(34, 20, 14) + 1  # 35
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: 1w EMA34 direction
        uptrend = curr_close > ema_34_1w_aligned[i]
        downtrend = curr_close < ema_34_1w_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = curr_close > donchian_high[i]
        breakout_down = curr_close < donchian_low[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above Donchian high AND uptrend
            if breakout_up and uptrend:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Short: Breakdown below Donchian low AND downtrend
            elif breakout_down and downtrend:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: exit when price drops 2.5*ATR below entry
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit on breakdown below Donchian low (reversal signal)
            elif breakout_down:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Stoploss: exit when price rises 2.5*ATR above entry
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit on breakout above Donchian high (reversal signal)
            elif breakout_up:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
    
    return signals