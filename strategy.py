#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high, 1w EMA34 is rising, and volume > 1.5x 20-bar average
# Short when price breaks below Donchian(20) low, 1w EMA34 is falling, and volume > 1.5x 20-bar average
# Uses 1w EMA34 for higher-timeframe trend to avoid whipsaws in ranging markets
# Volume spike confirms breakout momentum
# Discrete position sizing (0.25) to minimize fee churn
# Designed for low trade frequency (~10-25/year on 1d) to minimize fee drag
# Works in bull (breakouts with rising 1w EMA34) and bear (breakdowns with falling 1w EMA34)

name = "1d_Donchian20_Volume_1wEMA34_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on 1d
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    # Calculate EMA34 on 1w close
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align to 1d timeframe (completed 1w bar only)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation (1.5x 20-period average) on 1d
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(20, 34, 20) + 1  # Donchian(20) + EMA34(1w) + volume MA(20) warmup + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Donchian high, 1w EMA34 rising, volume spike
            if (close[i] > donchian_high[i] and 
                ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price < Donchian low, 1w EMA34 falling, volume spike
            elif (close[i] < donchian_low[i] and 
                  ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Donchian low or 1w EMA34 starts falling
            if (close[i] < donchian_low[i] or 
                ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > Donchian high or 1w EMA34 starts rising
            if (close[i] > donchian_high[i] or 
                ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals