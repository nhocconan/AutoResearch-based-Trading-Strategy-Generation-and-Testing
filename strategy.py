#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout + weekly volume confirmation + ATR stoploss
# - Donchian(20) breakout captures momentum in both bull and bear markets
# - Weekly volume confirmation ensures breakouts have institutional participation
# - ATR-based stoploss limits drawdown during volatile periods
# - Discrete position sizing (±0.25) minimizes fee churn
# - Target: 15-25 trades/year (60-100 total over 4 years) to stay within fee limits
# - Works in bull markets (breakouts to upside) and bear markets (breakouts to downside)
# - Weekly timeframe avoids noise while capturing major participation

name = "1d_donchian_breakout_weekly_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    # Load weekly data ONCE before loop for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute weekly volume confirmation (20-period average)
    volume_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    # Pre-compute Donchian channels (20-period) on 1d
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(60, n):  # Start after 60-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume data
        close_current = close[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_up = close_current > donchian_high[i-1]  # Break above previous period high
        breakout_down = close_current < donchian_low[i-1]  # Break below previous period low
        
        # Weekly volume confirmation: current volume > 1.5x 20-period weekly average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # ATR-based stoploss (2.0 * ATR)
        stop_distance = 2.0 * atr[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: upward Donchian breakout + volume confirmation
        if breakout_up and vol_confirm:
            enter_long = True
        
        # Short: downward Donchian breakout + volume confirmation
        if breakout_down and vol_confirm:
            enter_short = True
        
        # Exit conditions: stoploss hit or opposite breakout
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if stoploss hit OR downward breakout occurs
            if close_current <= entry_price - stop_distance:
                exit_long = True
            elif breakout_down:  # Opposite breakout
                exit_long = True
        elif position == -1:
            # Exit short if stoploss hit OR upward breakout occurs
            if close_current >= entry_price + stop_distance:
                exit_short = True
            elif breakout_up:  # Opposite breakout
                exit_short = True
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            entry_price = close_current
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            entry_price = close_current
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals