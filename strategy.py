#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian Breakout + 1w EMA Trend Filter + Volume Spike
# Uses weekly trend direction to filter daily breakouts, avoiding false signals in counter-trend moves
# Volume > 1.5x average confirms breakout strength
# Position size 0.25 to manage drawdown in volatile markets
# Target: 30-100 total trades over 4 years (7-25/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1h data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Donchian channel (20-period) on daily
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 40  # for Donchian and volume calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade in direction of weekly EMA
        if price > ema_1w_aligned[i]:
            # Only allow longs in uptrend
            if position == 0:
                # Long: price breaks above Donchian high with volume filter
                if price > donchian_high[i] and vol > 1.5 * avg_vol[i]:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Stay long
                signals[i] = position_size
            elif position == -1:
                # Exit short: flip to flat
                position = 0
                signals[i] = 0.0
        else:
            # Only allow shorts in downtrend
            if price < ema_1w_aligned[i]:
                if position == 0:
                    # Short: price breaks below Donchian low with volume filter
                    if price < donchian_low[i] and vol > 1.5 * avg_vol[i]:
                        position = -1
                        signals[i] = -position_size
                    else:
                        signals[i] = 0.0
                elif position == -1:
                    # Stay short
                    signals[i] = -position_size
                elif position == 1:
                    # Exit long: flip to flat
                    position = 0
                    signals[i] = 0.0
            else:
                # Price exactly at EMA (rare), stay flat
                signals[i] = 0.0
                if position != 0:
                    position = 0
    
    return signals

name = "1d_Donchian_WeeklyEMA_Volume_Filter"
timeframe = "1d"
leverage = 1.0