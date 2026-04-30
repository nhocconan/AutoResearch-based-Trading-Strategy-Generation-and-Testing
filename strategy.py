#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian breakouts capture momentum bursts in both bull and bear markets.
# 1d EMA50 provides primary trend filter: only long when price > EMA50, short when price < EMA50.
# Volume spike (>2.0x 20-period average) confirms breakout strength and reduces false signals.
# ATR-based stoploss (2.5x ATR) manages risk in volatile markets.
# Discrete sizing 0.25 minimizes fee churn. Target: 12-37 trades/year (50-150 total over 4 years).
# Works in bull via breakouts with uptrend, in bear via breakdowns with downtrend.

name = "12h_Donchian20_Breakout_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Calculate ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = max(donchian_window, 20, 50, 14)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_upper = upper_channel[i]
        curr_lower = lower_channel[i]
        curr_ema_50 = ema_50_aligned[i]
        curr_atr = atr[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish breakout: price breaks above upper Donchian channel AND price > 1d EMA50 (uptrend)
                if curr_close > curr_upper and curr_close > curr_ema_50:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    atr_stop = curr_close - (2.5 * curr_atr)
                # Bearish breakdown: price breaks below lower Donchian channel AND price < 1d EMA50 (downtrend)
                elif curr_close < curr_lower and curr_close < curr_ema_50:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    atr_stop = curr_close + (2.5 * curr_atr)
        
        elif position == 1:  # Long position
            # Exit when price drops below ATR stoploss OR breaks below lower Donchian channel
            if curr_close < atr_stop or curr_close < curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail the stoploss upward
                atr_stop = max(atr_stop, curr_close - (2.5 * curr_atr))
        
        elif position == -1:  # Short position
            # Exit when price rises above ATR stoploss OR breaks above upper Donchian channel
            if curr_close > atr_stop or curr_close > curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail the stoploss downward
                atr_stop = min(atr_stop, curr_close + (2.5 * curr_atr))
    
    return signals