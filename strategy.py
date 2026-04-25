#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike
Hypothesis: On daily timeframe, use 20-period Donchian channel breakouts filtered by 1-week EMA34 trend and volume spike confirmation. Designed for very low trade frequency (7-25/year) to minimize fee drag. Works in both bull and bear markets by only taking breakouts in direction of higher timeframe trend. Includes volatility-based position sizing to reduce drawdown during choppy periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for volatility-based position sizing (using daily data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily Donchian channel (20-period breakout)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # 1-week EMA34 for trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: current volume > 2.0 * 50-period average (very strict to minimize trades)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian (20) + vol MA (50) + ATR (14)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Volatility filter: only trade when ATR is reasonable (avoid extreme chop)
        atr_ma = pd.Series(atr).rolling(window=30, min_periods=30).mean().values
        if i >= 30 and not np.isnan(atr_ma[i]):
            vol_filter = atr[i] < (atr_ma[i] * 2.5)  # Avoid extremely high volatility periods
        else:
            vol_filter = True
        
        if position == 0:
            # Look for entry signals with strict confluence
            # Long: price breaks above Donchian high + uptrend + volume spike + vol filter
            long_entry = (curr_close > donchian_high[i]) and \
                         (curr_close > ema_34_1w_aligned[i]) and \
                         volume_spike[i] and vol_filter
            
            # Short: price breaks below Donchian low + downtrend + volume spike + vol filter
            short_entry = (curr_close < donchian_low[i]) and \
                          (curr_close < ema_34_1w_aligned[i]) and \
                          volume_spike[i] and vol_filter
            
            if long_entry:
                # Position size inversely proportional to volatility (normalize by ATR)
                vol_scalar = min(1.5, max(0.5, atr_ma[i] / atr[i]) if i >= 30 and not np.isnan(atr_ma[i]) else 1.0)
                signal_size = 0.25 * vol_scalar
                signal_size = min(0.35, max(0.15, signal_size))  # Clamp between 0.15 and 0.35
                signals[i] = signal_size
                position = 1
                entry_price = curr_close
            elif short_entry:
                vol_scalar = min(1.5, max(0.5, atr_ma[i] / atr[i]) if i >= 30 and not np.isnan(atr_ma[i]) else 1.0)
                signal_size = 0.25 * vol_scalar
                signal_size = min(0.35, max(0.15, signal_size))
                signals[i] = -signal_size
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on Donchian low break or trend reversal
            if curr_close < donchian_low[i] or curr_close < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short position: exit on Donchian high break or trend reversal
            if curr_close > donchian_high[i] or curr_close > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0