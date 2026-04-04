#!/usr/bin/env python3
"""
exp_6703_4h_donchian20_12h_hma_v1
Hypothesis: 4h Donchian channel breakout with 12h HMA trend filter and volume confirmation.
In ranging markets (ADX < 20), fade Donchian extremes toward 20-period EMA.
In trending markets (ADX > 25), breakout in direction of 12h HMA trend.
ATR-based stoploss and time-based exit to limit drawdown. Designed for 4h to capture
medium-term swings while keeping trade frequency low (target: 20-50/year).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6703_4h_donchian20_12h_hma_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 20
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 6  # ~1 day (4h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for HMA trend
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA (Hull Moving Average) for trend
    def hull_moving_average(series, period):
        if len(series) < period:
            return np.full_like(series, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = pd.Series(series).ewm(span=half_period, adjust=False).mean()
        wma_full = pd.Series(series).ewm(span=period, adjust=False).mean()
        hma = 2 * wma_half - wma_full
        hma = pd.Series(hma).ewm(span=sqrt_period, adjust=False).mean()
        return hma.values
    
    hma_12h = hull_moving_average(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # EMA for mean reversion target
    ema_20 = pd.Series(close).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # ADX for regime detection
    plus_dm = pd.Series(np.where((high - pd.Series(high).shift(1)) > (pd.Series(low).shift(1) - low), 
                                 np.maximum(high - pd.Series(high).shift(1), 0), 0))
    minus_dm = pd.Series(np.where((pd.Series(low).shift(1) - low) > (high - pd.Series(high).shift(1)), 
                                  np.maximum(pd.Series(low).shift(1) - low, 0), 0))
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - pd.Series(close).shift(1)))
    tr3 = pd.Series(np.abs(low - pd.Series(close).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_raw = tr.ewm(span=ADX_PERIOD, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / atr_raw)
    minus_di = 100 * (minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / atr_raw)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(span=ADX_PERIOD, adjust=False).mean().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, ADX_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Determine market regime
        is_trending = adx[i] > ADX_TREND_THRESHOLD if not np.isnan(adx[i]) else False
        is_ranging = adx[i] < ADX_RANGE_THRESHOLD if not np.isnan(adx[i]) else False
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine trend direction from 12h HMA
        hma_trend_up = hma_12h_aligned[i] > hma_12h_aligned[i-1] if i > 0 and not np.isnan(hma_12h_aligned[i-1]) else False
        hma_trend_down = hma_12h_aligned[i] < hma_12h_aligned[i-1] if i > 0 and not np.isnan(hma_12h_aligned[i-1]) else False
        
        # Mean reversion signals (in ranging market)
        long_mean_revert = is_ranging and (close[i] <= donchian_low[i]) and vol_confirmed
        short_mean_revert = is_ranging and (close[i] >= donchian_high[i]) and vol_confirmed
        
        # Breakout signals (in trending market)
        long_breakout = is_trending and (close[i] > donchian_high[i]) and hma_trend_up and vol_confirmed
        short_breakout = is_trending and (close[i] < donchian_low[i]) and hma_trend_down and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_mean_revert or long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_mean_revert or short_breakout:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals