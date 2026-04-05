#!/usr/bin/env python3
"""
Experiment #9362: 12h Donchian breakout + 1d trend filter + volume confirmation + ATR stoploss.
Hypothesis: 12h Donchian breaks capture multi-day trends; 1d EMA filter ensures alignment with daily trend; volume confirms institutional participation. Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing trends in both bull and bear markets.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np

name = "exp_9362_12h_donchian20_1d_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
TREND_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.2

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = np.full_like(close, np.nan, dtype=np.float64)
    if len(close) > 0:
        tr_mean = np.nanmean(tr[:period]) if np.any(~np.isnan(tr[:period])) else np.nan
        atr[period-1] = tr_mean
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = np.full_like(close_1d, np.nan, dtype=np.float64)
    if len(close_1d) > 0:
        ema_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (TREND_PERIOD + 1)) + (ema_1d[i-1] * (TREND_PERIOD - 1) / (TREND_PERIOD + 1))
    
    # Price relative to 1d EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_1d > ema_1d, 1, 
                     np.where(close_1d < ema_1d, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate LTF indicators (12h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = np.full_like(high, np.nan, dtype=np.float64)
    donchian_low = np.full_like(low, np.nan, dtype=np.float64)
    for i in range(DONCHIAN_PERIOD-1, len(high)):
        donchian_high[i] = np.max(high[i-DONCHIAN_PERIOD+1:i+1])
        donchian_low[i] = np.min(low[i-DONCHIAN_PERIOD+1:i+1])
    
    # Volume moving average
    volume_ma = np.full_like(volume, np.nan, dtype=np.float64)
    for i in range(VOLUME_MA_PERIOD-1, len(volume)):
        volume_ma[i] = np.mean(volume[i-VOLUME_MA_PERIOD+1:i+1])
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]):
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
        
        # Determine market bias from 1d EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d price above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d price below EMA50
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1] if i > 0 and not np.isnan(donchian_high[i-1]) else False
        short_breakout = close[i] < donchian_low[i-1] if i > 0 and not np.isnan(donchian_low[i-1]) else False
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if i >= VOLUME_MA_PERIOD-1 and not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bull_bias and long_breakout and volume_confirmed
        short_entry = bear_bias and short_breakout and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals