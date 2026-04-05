#!/usr/bin/env python3
"""
Experiment #7919: 6-hour Elder Ray Power with 12h/1d regime filter and volume confirmation.
Hypothesis: Bull Power (high - EMA13) and Bear Power (EMA13 - low) capture institutional buying/selling pressure.
In trending regimes (12h ADX > 25), trade in direction of power; in ranging regimes (12h ADX < 20), fade extremes.
Uses 1d close > SMA50 for bull market bias, < SMA50 for bear bias to adapt to market cycles.
Volume confirmation ensures institutional participation. Target: 75-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7919_6h_elder_ray_12h_adx_1d_bias_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
EAR_LENGTH = 13
ADX_LEN = 14
ADX_TREND_THRESH = 25
ADX_RANGE_THRESH = 20
VOL_MA_LEN = 20
VOL_THRESH = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULT = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h ADX for regime detection
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = pd.Series(high_12h - np.roll(high_12h, 1))
    down_move = pd.Series(np.roll(low_12h, 1) - low_12h)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_ma = pd.Series(tr_12h).ewm(span=ADX_LEN, adjust=False, min_periods=ADX_LEN).mean().values
    plus_dm_ma = pd.Series(plus_dm).ewm(span=ADX_LEN, adjust=False, min_periods=ADX_LEN).mean().values
    minus_dm_ma = pd.Series(minus_dm).ewm(span=ADX_LEN, adjust=False, min_periods=ADX_LEN).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_ma / tr_ma
    minus_di = 100 * minus_dm_ma / tr_ma
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=ADX_LEN, adjust=False, min_periods=ADX_LEN).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate 1d bias: close > SMA50 = bull market, < SMA50 = bear market
    close_1d = df_1d['close'].values
    sma_1d_50 = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_50)
    bull_market_bias = close_1d > sma_1d_50  # 1d close above 50 SMA
    bull_market_bias_aligned = align_htf_to_ltf(prices, df_1d, bull_market_bias)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13 = pd.Series(close).ewm(span=EAR_LENGTH, adjust=False, min_periods=EAR_LENGTH).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOL_MA_LEN, min_periods=VOL_MA_LEN).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EAR_LENGTH, ADX_LEN, VOL_MA_LEN, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(adx_aligned[i]) or np.isnan(bull_market_bias_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Regime detection from 12h ADX
        trending = adx_aligned[i] > ADX_TREND_THRESH
        ranging = adx_aligned[i] < ADX_RANGE_THRESH
        
        # Market bias from 1d SMA50
        bull_bias = bull_market_bias_aligned[i]  # True if bull market bias
        bear_bias = not bull_market_bias_aligned[i]  # True if bear market bias
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOL_THRESH) if not np.isnan(volume_ma[i]) else False
        
        # Elder Ray signals
        bp_signal = bull_power[i] > 0  # Positive bull power = buying pressure
        bp_signal_strength = bull_power[i]  # For potential filtering
        bear_signal = bear_power[i] > 0  # Positive bear power = selling pressure
        bear_signal_strength = bear_power[i]
        
        # Entry logic: adapt to regime and market bias
        if position == 0:
            if trending:
                # In trending markets, trade with the power and bias
                long_entry = bp_signal and bull_bias and volume_confirmed
                short_entry = bear_signal and bear_bias and volume_confirmed
            elif ranging:
                # In ranging markets, fade extreme power readings
                # Long when bear power is exhausted (weak selling) in bull bias
                # Short when bull power is exhausted (weak buying) in bear bias
                long_entry = (bear_power[i] < 0) and bull_bias and volume_confirmed  # Bear power negative = selling weakening
                short_entry = (bull_power[i] < 0) and bear_bias and volume_confirmed   # Bull power negative = buying weakening
            else:
                # Transition period: neutral
                long_entry = False
                short_entry = False
            
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULT * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULT * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals