#!/usr/bin/env python3
"""
Experiment #6449: 4h Donchian(20) breakout with 1d EMA(21) trend filter and volume confirmation.
Hypothesis: Price breaking above/below 20-period Donchian channel on 4h timeframe,
            confirmed by 1d EMA(21) direction and above-average volume,
            provides profitable breakout signals in both bull and bear markets.
            Uses ATR-based stoploss for risk control.
Timeframe: 4h
HTF: 1d for EMA filter
Position size: 0.25 (discrete level to minimize fee churn)
Max trades target: 75-200 total over 4 years (19-50/year)
"""

# === MODULE IMPORTS ===
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# === STRATEGY PARAMETERS ===
name = "exp_6449_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Donchian channel period
DONCHIAN_PERIOD = 20
# EMA period on 1d timeframe
EMA_PERIOD = 21
# Volume confirmation: volume > average volume
VOLUME_LOOKBACK = 20
# ATR for stoploss
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
# Position size
POSITION_SIZE = 0.25

def generate_signals(prices):
    """
    Generate trading signals for BTC/ETH/SOL USDT-M perpetual futures.
    
    Parameters:
    prices: DataFrame with columns ['open_time', 'open', 'high', 'low', 'close', 'volume', ...]
            Index is DatetimeIndex already (no conversion needed)
    
    Returns:
    np.ndarray: signal values from -1.0 to 1.0 (position size as fraction of capital)
    """
    n = len(prices)
    if n < 100:  # Need sufficient data for indicators
        return np.zeros(n)
    
    # === PRECOMPUTE INDICATORS BEFORE LOOP ===
    # Primary timeframe indicators (4h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels: highest high and lowest low over DONCHIAN_PERIOD
    # Use pandas rolling with min_periods to avoid look-ahead
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_lower = low_series.rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Average volume for confirmation
    volume_series = pd.Series(volume)
    avg_volume = volume_series.rolling(window=VOLUME_LOOKBACK, min_periods=VOLUME_LOOKBACK).mean().values
    
    # ATR for stoploss calculation
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = tr1.iloc[0]  # First TR is just high-low
    tr3.iloc[0] = tr1.iloc[0]
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=ATR_PERIOD, min_periods=ATR_PERIOD).mean().values
    
    # === MULTI-TIMEFRAME: 1d EMA trend filter ===
    # Load 1d data ONCE before loop (critical for performance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < EMA_PERIOD:
        return np.zeros(n)
    
    # Calculate EMA on 1d close prices
    close_1d = pd.Series(df_1d['close'].values)
    ema_1d = close_1d.ewm(span=EMA_PERIOD, min_periods=EMA_PERIOD, adjust=False).mean().values
    
    # Align 1d EMA to 4h timeframe (with shift(1) to avoid look-ahead)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === SIGNAL GENERATION LOOP ===
    signals = np.zeros(n)
    position = 0  # Track current position: 0=flat, 1=long, -1=short
    entry_price = 0.0
    
    # Start loop after warmup period (need enough data for all indicators)
    start_idx = max(DONCHIAN_PERIOD, VOLUME_LOOKBACK, ATR_PERIOD, EMA_PERIOD)
    
    for i in range(start_idx, n):
        # Skip if any indicator is not ready (NaN)
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(atr[i]) or 
            np.isnan(ema_1d_aligned[i])):
            continue
        
        # Current price action
        curr_close = close[i]
        curr_volume = volume[i]
        
        # === ENTRY CONDITIONS ===
        # Long: price breaks above Donchian upper + volume > average + 1d EMA rising
        long_breakout = curr_close > donchian_upper[i]
        long_volume = curr_volume > avg_volume[i]
        # 1d EMA trending up: current EMA > previous EMA
        long_ema_trend = ema_1d_aligned[i] > ema_1d_aligned[i-1] if i > 0 else False
        
        # Short: price breaks below Donchian lower + volume > average + 1d EMA falling
        short_breakout = curr_close < donchian_lower[i]
        short_volume = curr_volume > avg_volume[i]
        # 1d EMA trending down: current EMA < previous EMA
        short_ema_trend = ema_1d_aligned[i] < ema_1d_aligned[i-1] if i > 0 else False
        
        # === EXIT CONDITIONS ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Stoploss: price drops below entry - ATR_MULTIPLIER * ATR
            if curr_close < entry_price - ATR_MULTIPLIER * atr[i]:
                exit_signal = True
            # Optional: exit on Donchian lower break (contrarian exit)
            elif curr_close < donchian_lower[i]:
                exit_signal = True
                
        elif position == -1:  # Short position
            # Stoploss: price rises above entry + ATR_MULTIPLIER * ATR
            if curr_close > entry_price + ATR_MULTIPLIER * atr[i]:
                exit_signal = True
            # Optional: exit on Donchian upper break (contrarian exit)
            elif curr_close > donchian_upper[i]:
                exit_signal = True
        
        # === SIGNAL LOGIC ===
        if exit_signal:
            signals[i] = 0.0  # Close position
            position = 0
        elif position == 0:  # Only look for new entries when flat
            # Long entry conditions
            if long_breakout and long_volume and long_ema_trend:
                signals[i] = POSITION_SIZE
                position = 1
                entry_price = curr_close
            # Short entry conditions
            elif short_breakout and short_volume and short_ema_trend:
                signals[i] = -POSITION_SIZE
                position = -1
                entry_price = curr_close
        else:
            # Holding position - maintain signal
            signals[i] = POSITION_SIZE if position == 1 else -POSITION_SIZE
    
    return signals