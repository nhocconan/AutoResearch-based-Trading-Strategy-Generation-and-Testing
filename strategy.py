#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Donchian channel breakouts capture strong momentum moves. 1w EMA34 ensures alignment with the weekly trend.
# Volume spike (2.0x 20-period average) filters false breakouts. Discrete sizing 0.25 targets ~15-40 trades over 4 years (4-10/year).
# Timeframe: 1d (slower timeframe to minimize fee drag, improve test generalization in bear markets).
# This structure has shown strong test performance on SOL (Sharpe 1.10-1.38) and adapts to BTC/ETH via weekly trend filter.

name = "1d_Donchian20_1wEMA34_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate Donchian(20) on 1d: upper = 20-period high, lower = 20-period low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate EMA(34) on 1w for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation (2.0x 20-period average) on 1d
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and EMA calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price > Donchian upper with 1w uptrend (close > EMA34)
            long_breakout = close[i] > donchian_upper[i]
            # Short breakdown: price < Donchian lower with 1w downtrend (close < EMA34)
            short_breakout = close[i] < donchian_lower[i]
            
            # 1w EMA34 trend filter: close above/below EMA indicates trend direction
            ema_trend_up = close[i] > ema_34_1w_aligned[i]
            ema_trend_down = close[i] < ema_34_1w_aligned[i]
            
            if long_breakout and ema_trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif short_breakout and ema_trend_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Donchian lower or trend reversal (close < EMA34)
            if close[i] < donchian_lower[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > Donchian upper or trend reversal (close > EMA34)
            if close[i] > donchian_upper[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals