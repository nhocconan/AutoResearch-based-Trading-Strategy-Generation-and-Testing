#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w volume spike and ATR regime filter.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for volume average calculation (to avoid intraday noise) and trend confirmation.
- Donchian channels: identifies breakouts from 20-day price channels.
- Entry: Long when price breaks above upper Donchian(20) AND volume > 1.5 * 20-period weekly average volume AND price > 50-week EMA (bullish regime).
         Short when price breaks below lower Donchian(20) AND volume > 1.5 * 20-period weekly average volume AND price < 50-week EMA (bearish regime).
- Exit: Opposite Donchian breakout (price crosses back below upper for longs, above lower for shorts) OR ATR-based stoploss (2 * ATR(10) from entry).
- Signal size: 0.25 discrete to minimize fee drag.
- Donchian breakouts capture strong momentum moves after consolidation.
- Weekly volume confirmation ensures breakout legitimacy with institutional participation.
- Weekly EMA filter ensures trading in direction of higher timeframe trend.
- Works in both bull and bear markets by adapting to trend regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def atr(high, low, close, period):
    """Calculate Average True Range with proper min_periods."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_values = tr.ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr_values

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need sufficient weeks for Donchian
        return np.zeros(n)
    
    # Weekly upper and lower Donchian channels (20-period)
    donch_high_20 = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20)
    
    # Calculate weekly volume average for confirmation (20-period)
    if len(df_1w) < 20:
        return np.zeros(n)
    
    vol_ma_20_1w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # Calculate weekly 50-period EMA for trend filter
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily ATR for stoploss (10-period)
    if len(prices) < 10:
        return np.zeros(n)
    
    atr_10 = atr(high, low, close, 10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for stoploss
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for Donchian/volume, 50 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or
            np.isnan(vol_ma_20_1w_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions
        if position != 0:
            # Exit conditions: opposite Donchian breakout OR ATR stoploss
            exit_signal = False
            
            # Opposite Donchian breakout
            if position == 1:  # Long position
                if curr_close < donch_high_20_aligned[i]:
                    exit_signal = True
                # ATR-based stoploss: 2 * ATR(10) below entry
                elif curr_close < entry_price - 2.0 * atr_10[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if curr_close > donch_low_20_aligned[i]:
                    exit_signal = True
                # ATR-based stoploss: 2 * ATR(10) above entry
                elif curr_close > entry_price + 2.0 * atr_10[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # Entry conditions
        if position == 0:
            # Donchian breakout signals
            breakout_up = curr_high >= donch_high_20_aligned[i] and prev_close < donch_high_20_aligned[i-1]
            breakout_down = curr_low <= donch_low_20_aligned[i] and prev_close > donch_low_20_aligned[i-1]
            
            # Volume confirmation: current volume > 1.5 * 20-period weekly average volume
            volume_confirm = curr_volume > 1.5 * vol_ma_20_1w_aligned[i] if not np.isnan(vol_ma_20_1w_aligned[i]) else False
            
            # Trend filter: price relative to weekly 50 EMA
            trend_filter_up = curr_close > ema_50_1w_aligned[i]
            trend_filter_down = curr_close < ema_50_1w_aligned[i]
            
            if breakout_up and volume_confirm and trend_filter_up:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif breakout_down and volume_confirm and trend_filter_down:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wVolumeSpike_EMA50Trend_v1"
timeframe = "1d"
leverage = 1.0