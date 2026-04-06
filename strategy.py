#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h price action with 1-week trend filter and volume confirmation
# Enter long when: price breaks above 12h Donchian(20) high, weekly EMA(20) rising, volume > 1.5x 20-period average
# Enter short when: price breaks below 12h Donchian(20) low, weekly EMA(20) falling, volume > 1.5x 20-period average
# Exit with ATR-based stoploss (2x ATR) or when price returns to Donchian midpoint
# Designed for low frequency (15-30 trades/year) with clear trend signals that work in both bull and bear markets

name = "12h_donchian20_weekly_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 12h ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = tr1.iloc[0]
    tr3.iloc[0] = tr1.iloc[0]
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Weekly EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    ema_20_1w_prev = np.roll(ema_20_1w_aligned, 1)
    ema_20_1w_prev[0] = ema_20_1w_aligned[0]
    ema_rising = ema_20_1w_aligned > ema_20_1w_prev
    ema_falling = ema_20_1w_aligned < ema_20_1w_prev
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_threshold[i]) or
            np.isnan(atr[i])):
            if position != 0:
                # Maintain position with stoploss check
                if position == 1 and low[i] < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and high[i] > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: stoploss hit OR price returns to Donchian midpoint
            if low[i] < entry_price - 2.0 * atr[i] or abs(close[i] - donchian_mid[i]) < 0.1 * (donchian_high[i] - donchian_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: stoploss hit OR price returns to Donchian midpoint
            if high[i] > entry_price + 2.0 * atr[i] or abs(close[i] - donchian_mid[i]) < 0.1 * (donchian_high[i] - donchian_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries with trend and volume confirmation
            if volume[i] > volume_threshold[i]:
                # Long breakout: price above Donchian high with rising weekly EMA
                if close[i] > donchian_high[i] and ema_rising[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakout: price below Donchian low with falling weekly EMA
                elif close[i] < donchian_low[i] and ema_falling[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals