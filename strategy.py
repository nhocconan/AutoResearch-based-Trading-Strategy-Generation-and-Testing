#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian(20) Breakout + Volume Confirmation + ADX Trend Filter
# Hypothesis: Donchian breakouts capture strong momentum moves. Volume confirms
# institutional participation. ADX ensures we only trade in trending markets,
# avoiding whipsaws in ranges. Works in bull/bear by following breakout direction.
# Target: 12-37 trades/year on 12h timeframe to minimize fee drag.
name = "12h_donchian20_volume_adx_v1"
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
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # ADX (14-period) for trend strength
    # TR = max(high-low, |high-close_prev|, |low-close_prev|)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    
    # DX and ADX
    dx = np.zeros(n)
    dx_sum = pd.Series(np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100).rolling(window=14, min_periods=14).mean().values
    adx = dx_sum  # ADX is smoothed DX
    
    # Get daily trend filter (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_12h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(adx[i]) or 
            np.isnan(daily_ema_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band or trend turns bearish
            if close[i] < lowest_low[i] or close[i] < daily_ema_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band or trend turns bullish
            if close[i] > highest_high[i] or close[i] > daily_ema_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Only trade in strong trending markets (ADX > 25)
            if adx[i] > 25:
                # Enter long: price breaks above Donchian upper band + volume spike + bullish daily trend
                if close[i] > highest_high[i] and volume_spike[i] and close[i] > daily_ema_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short: price breaks below Donchian lower band + volume spike + bearish daily trend
                elif close[i] < lowest_low[i] and volume_spike[i] and close[i] < daily_ema_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals