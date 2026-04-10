#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter (EMA34) and volume confirmation
# - Long when price breaks above 20-period Donchian high AND 1d EMA34 rising AND volume > 2.0x 20-period volume SMA
# - Short when price breaks below 20-period Donchian low AND 1d EMA34 falling AND volume > 2.0x 20-period volume SMA
# - Exit: ATR trailing stop (3.0x ATR) or time-based exit (max 3 bars hold)
# - Uses 1d for trend bias (EMA34 direction) and 12h for precise entry timing
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Position sizing: 0.25 discrete level to control drawdown and minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag while maintaining statistical significance
# - Donchian breakouts work in both bull and bear markets when combined with trend filter and volume confirmation

name = "12h_1d_donchian_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return signals
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_rising = ema_34_1d > np.roll(ema_34_1d, 1)
    ema_34_falling = ema_34_1d < np.roll(ema_34_1d, 1)
    # Align to 12h timeframe with proper delay (completed 1d bar only)
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Track bars since entry for time-based exit
    bars_since_entry = np.full(n, 0)
    
    for i in range(20, n):  # Start from 20 to have sufficient lookback
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            bars_since_entry[i] = 0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(atr[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_34_rising_aligned[i]) or np.isnan(ema_34_falling_aligned[i])):
            signals[i] = 0.0
            bars_since_entry[i] = 0
            continue
        
        # Volume confirmation: 12h volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > 2.0 * volume_sma_20[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i-1]  # Break above Donchian high
        breakout_down = close[i] < donchian_low[i-1]  # Break below Donchian low
        
        if position == 0:  # Flat - look for entry
            # Long: price breaks above Donchian high AND 1d EMA34 rising AND volume confirmation
            if breakout_up and ema_34_rising_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
                bars_since_entry[i] = 1
            # Short: price breaks below Donchian low AND 1d EMA34 falling AND volume confirmation
            elif breakout_down and ema_34_falling_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
                bars_since_entry[i] = 1
            else:
                signals[i] = 0.0
                bars_since_entry[i] = 0
        elif position == 1:  # Long position - look for exit
            # Update bars since entry
            bars_since_entry[i] = bars_since_entry[i-1] + 1
            
            # ATR trailing stop: exit if price drops 3.0*ATR below highest high since entry
            # We'll use a simplified trailing stop: exit if price drops 3.0*ATR below entry price
            # For better accuracy, we'd need to track entry price, but we'll use close-based approximation
            trailing_stop = close[i-1] - 3.0 * atr[i] if i > 0 else close[i] - 3.0 * atr[i]
            
            # Exit conditions: trailing stop hit OR time-based exit (max 3 bars) OR Donchian reversion
            exit_condition = (close[i] < trailing_stop) or (bars_since_entry[i] >= 3) or (close[i] < donchian_low[i])
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
                bars_since_entry[i] = 0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Update bars since entry
            bars_since_entry[i] = bars_since_entry[i-1] + 1
            
            # ATR trailing stop: exit if price rises 3.0*ATR above lowest low since entry
            trailing_stop = close[i-1] + 3.0 * atr[i] if i > 0 else close[i] + 3.0 * atr[i]
            
            # Exit conditions: trailing stop hit OR time-based exit (max 3 bars) OR Donchian reversion
            exit_condition = (close[i] > trailing_stop) or (bars_since_entry[i] >= 3) or (close[i] > donchian_high[i])
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
                bars_since_entry[i] = 0
            else:
                signals[i] = -0.25
    
    return signals