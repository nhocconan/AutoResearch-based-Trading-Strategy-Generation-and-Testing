#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout direction + 1d EMA50 trend filter + volume confirmation (>1.5x 20-bar avg)
# Uses 4h/1d for signal direction (HTF), 1h only for entry timing precision.
# Session filter (08-20 UTC) to reduce noise trades. Discrete position size 0.20.
# Target: 15-37 trades/year via tight conditions suitable for BTC/ETH in both bull and bear markets.

name = "1h_Donchian20_4hTrend_1dEMA50_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period) using previous bar's OHLC
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_series_4h = pd.Series(high_4h)
    low_series_4h = pd.Series(low_4h)
    
    donchian_upper_4h = high_series_4h.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower_4h = low_series_4h.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align 4h Donchian to 1h timeframe (completed 4h candles only)
    donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 1h timeframe (completed 1d candles only)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: >1.5x 20-bar average volume on 1h
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20)  # Need sufficient history for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_upper_4h_aligned[i]) or np.isnan(donchian_lower_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        donchian_upper = donchian_upper_4h_aligned[i]
        donchian_lower = donchian_lower_4h_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long breakout: price breaks above 4h Donchian Upper AND price > 1d EMA50 (uptrend) AND volume spike
            if price > donchian_upper and price > ema_trend and vol_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short breakout: price breaks below 4h Donchian Lower AND price < 1d EMA50 (downtrend) AND volume spike
            elif price < donchian_lower and price < ema_trend and vol_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on opposite Donchian level touch
            # Exit on price < 4h Donchian Lower (opposite level touch)
            if price < donchian_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit on opposite Donchian level touch
            # Exit on price > 4h Donchian Upper (opposite level touch)
            if price > donchian_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals