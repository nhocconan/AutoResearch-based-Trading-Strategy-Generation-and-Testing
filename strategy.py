#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume spike + ATR stoploss
# Long when price breaks above Donchian upper + 12h EMA50 uptrend + volume > 2.0x 20-period avg
# Short when price breaks below Donchian lower + 12h EMA50 downtrend + volume > 2.0x 20-period avg
# Exit when price closes below Donchian lower (long) or above Donchian upper (short)
# Uses discrete position sizing (0.30) to balance return and drawdown.
# 12h EMA50 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (2.0x) targets ~30-50 trades/year on 4h timeframe to avoid overtrading.
# Donchian channels provide objective breakout levels with clear exit rules.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h Indicator: EMA50 ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === Donchian Channels (20-period) ===
    # Upper = highest high over past 20 periods
    # Lower = lowest low over past 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # track position: 1=long, -1=short, 0=flat
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20) + 5  # EMA50 + Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # === EXIT LOGIC ===
        # Close long if price closes below Donchian lower
        # Close short if price closes above Donchian upper
        if position == 1 and close[i] < donchian_lower[i]:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] > donchian_upper[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC ===
        # Only enter if flat
        if position == 0:
            # LONG: break above upper + uptrend + volume
            if (close[i] > donchian_upper[i]) and \
               (close[i] > ema_50_12h_aligned[i]) and vol_confirm:
                signals[i] = 0.30
                position = 1
            # SHORT: break below lower + downtrend + volume
            elif (close[i] < donchian_lower[i]) and \
                 (close[i] < ema_50_12h_aligned[i]) and vol_confirm:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4h_Donchian20_12hEMA50_Volume_Spike_v1"
timeframe = "4h"
leverage = 1.0