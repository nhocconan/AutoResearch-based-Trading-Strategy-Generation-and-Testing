#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout direction + 1d trend filter (EMA50) + volume spike
# Uses 4h for signal direction (breakout of 20-period channel) and 1d EMA50 for trend filter (avoid counter-trend trades)
# Volume spike (>2x 20-period average) confirms momentum. Discrete sizing 0.20 to minimize fee drag.
# Session filter (08-20 UTC) reduces noise trades. Target: 60-150 total trades over 4 years (15-37/year).
# Works in bull markets (breakouts with uptrend) and bear markets (breakouts with downtrend filtered by 1d EMA50).
# Focus on BTC/ETH as primary symbols with proven edge from Donchian + volume + trend confluence.

name = "1h_Donchian20_4hDir_1dEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - prices.index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h Donchian channels (20-period) - for direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    donchian_high_4h = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 20, 50, 20, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if indicators not ready
        if (np.isnan(donchian_high_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high_4h = donchian_high_4h_aligned[i]
        curr_donchian_low_4h = donchian_low_4h_aligned[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr_14[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Donchian break and 1d EMA50 trend filter
            if curr_volume_spike:
                # Bullish: Close breaks above 4h Donchian high + price above 1d EMA50 (uptrend)
                if curr_close > curr_donchian_high_4h and curr_close > curr_ema_50_1d:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish: Close breaks below 4h Donchian low + price below 1d EMA50 (downtrend)
                elif curr_close < curr_donchian_low_4h and curr_close < curr_ema_50_1d:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * curr_atr
            # Exit: Stoploss hit OR close drops below 4h Donchian low OR loses 1d uptrend
            if curr_low <= stop_loss or curr_close < curr_donchian_low_4h or curr_close < curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * curr_atr
            # Exit: Stoploss hit OR close rises above 4h Donchian high OR loses 1d downtrend
            if curr_high >= stop_loss or curr_close > curr_donchian_high_4h or curr_close > curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals