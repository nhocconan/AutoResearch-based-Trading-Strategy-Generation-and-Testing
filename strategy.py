#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 12-hour trend filter and volume confirmation
# Long when price breaks above Donchian upper (20) + 12h EMA trend up + volume > 1.5x average
# Short when price breaks below Donchian lower (20) + 12h EMA trend down + volume > 1.5x average
# Exit when price crosses Donchian middle or trend reverses
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 80-160 total trades over 4 years (20-40/year)

name = "4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
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
    
    # 12-hour data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12-hour EMA(20) for trend
    close_12h = df_12h['close'].values
    close_12h_s = pd.Series(close_12h)
    ema_12h = close_12h_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Volume average (20-period)
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian middle or trend reverses
            elif close[i] < donchian_middle[i] or ema_12h_aligned[i] < close_12h_s.ewm(span=20, adjust=False).mean().iloc[-1] if len(close_12h_s) > i//12 else ema_12h_aligned[i]:
                # Simplified: exit on middle line cross or price below Donchian middle
                if close[i] < donchian_middle[i]:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian middle or trend reverses
            elif close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume and trend confirmation
            volume_confirm = volume[i] > 1.5 * volume_ma[i]
            
            # Long: price breaks above Donchian upper + volume + 12h EMA up
            if close[i] > donchian_upper[i] and volume_confirm and ema_12h_aligned[i] > close_12h_s.ewm(span=20, adjust=False).mean().iloc[-1] if len(close_12h_s) > i//12 else ema_12h_aligned[i]:
                # Simplified trend check: current 12h EMA > previous value
                if i >= 12:  # Need at least one 12h bar behind
                    ema_prev = ema_12h_aligned[i-12] if i-12 >= 0 else ema_12h_aligned[0]
                    if ema_12h_aligned[i] > ema_prev:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                else:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
            # Short: price breaks below Donchian lower + volume + 12h EMA down
            elif close[i] < donchian_lower[i] and volume_confirm and ema_12h_aligned[i] < close_12h_s.ewm(span=20, adjust=False).mean().iloc[-1] if len(close_12h_s) > i//12 else ema_12h_aligned[i]:
                # Simplified trend check: current 12h EMA < previous value
                if i >= 12:
                    ema_prev = ema_12h_aligned[i-12] if i-12 >= 0 else ema_12h_aligned[0]
                    if ema_12h_aligned[i] < ema_prev:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
                else:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals