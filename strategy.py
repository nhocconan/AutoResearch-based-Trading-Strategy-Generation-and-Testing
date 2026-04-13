#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 1d ATR regime filter + volume confirmation
    # Long when: price breaks above 4h Donchian upper (20) AND ATR(14) < 1.5 * ATR(50) (low vol regime) AND volume > 1.5x 20-bar avg
    # Short when: price breaks below 4h Donchian lower (20) AND ATR(14) < 1.5 * ATR(50) AND volume > 1.5x 20-bar avg
    # Exit when: price crosses 4h Donchian midpoint
    # Uses discrete sizing (0.25) targeting 75-150 total trades over 4 years.
    # Donchian provides structure; ATR regime filter avoids high-vol whipsaws; volume confirms validity.
    # Works in bull (breakouts with trend) and bear (avoids false breakouts in ranging markets).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR for regime filter
    atr_period = 14
    atr_ma_period = 50
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    atr_ma = pd.Series(atr).ewm(span=atr_ma_period, adjust=False, min_periods=atr_ma_period).mean().values
    atr_ratio = atr / atr_ma  # Current ATR relative to longer-term average
    low_vol_regime = atr_ratio < 1.5  # Low volatility regime filter
    
    # Calculate 4h Donchian channels (20-period)
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(donchian_window, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(atr_ratio[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > donchian_high[i-1]  # break above previous Donchian high
        breakout_down = close[i] < donchian_low[i-1]  # break below previous Donchian low
        
        # Entry conditions with regime filter and volume confirmation
        long_entry = breakout_up and low_vol_regime[i] and volume_confirmed[i] and position != 1
        short_entry = breakout_down and low_vol_regime[i] and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close[i] < donchian_mid[i])
        exit_short = (position == -1 and close[i] > donchian_mid[i])
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "4h_donchian_atr_regime_volume_v1"
timeframe = "4h"
leverage = 1.0