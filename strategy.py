#!/usr/bin/env python3
"""
exp_6739_6h_elder_ray_regime_v1
Hypothesis: 6h Elder Ray (Bull/Bear Power) with ADX regime filter and EMA200 trend filter.
In trending markets (ADX>25): trade in direction of EMA200 using Elder Ray pullbacks.
In ranging markets (ADX<20): fade extreme Bull/Bear Power readings.
Volume confirmation on all entries.
Designed for 6h timeframe to capture swings with ~12-37 trades/year (50-150 total over 4 years).
Uses higher timeframe (12h) EMA200 for regime alignment to avoid whipsaws.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6739_6h_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
EMA_SHORT = 13
EMA_LONG = 21
EMA_200_PERIOD = 200
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 12  # ~3 days (6h bars)
POWER_THRESHOLD = 0.02  # 2% of price for extreme power

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Need enough data for EMA200
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for regime alignment
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA200 for trend filter
    close_12h = df_12h['close'].values
    ema_200_12h = pd.Series(close_12h).ewm(span=EMA_200_PERIOD, adjust=False, min_periods=EMA_200_PERIOD).mean().values
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMAs for Elder Ray
    ema_short = pd.Series(close).ewm(span=EMA_SHORT, adjust=False, min_periods=EMA_SHORT).mean().values
    ema_long = pd.Series(close).ewm(span=EMA_LONG, adjust=False, min_periods=EMA_LONG).mean().values
    
    # Elder Ray components
    bull_power = high - ema_long  # Bull Power: High - EMA(13)
    bear_power = low - ema_short  # Bear Power: Low - EMA(21)
    
    # ADX calculation
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_raw = pd.Series(tr).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values / atr_raw
    minus_di = 100 * pd.Series(minus_dm).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values / atr_raw
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # ATR for stoploss (separate from ADX ATR)
    tr_atr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr = pd.Series(tr_atr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(EMA_200_PERIOD, EMA_LONG, ADX_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_200_12h_aligned[i]):
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
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Regime determination using 12h EMA200 and ADX
        price_vs_ema200 = close[i] > ema_200_12h_aligned[i]
        is_trending = adx[i] > ADX_TREND_THRESHOLD
        is_ranging = adx[i] < ADX_RANGE_THRESHOLD
        
        # Extreme power thresholds (adaptive to price level)
        power_threshold_abs = close[i] * POWER_THRESHOLD
        
        # Initialize signal
        new_position = position
        
        if position == 0:  # Only look for new entries when flat
            if is_trending:
                # In trending markets: trade with EMA200 trend on Elder Ray pullbacks
                if price_vs_ema200:  # Uptrend
                    # Long on Bull Power weakness (pullback) with volume
                    if bull_power[i] < power_threshold_abs and bull_power[i] > -power_threshold_abs * 2 and vol_confirmed:
                        new_position = 1
                else:  # Downtrend
                    # Short on Bear Power strength (pullback) with volume
                    if bear_power[i] > -power_threshold_abs and bear_power[i] < power_threshold_abs * 2 and vol_confirmed:
                        new_position = -1
            elif is_ranging:
                # In ranging markets: fade extreme Elder Power readings
                if bull_power[i] > power_threshold_abs and vol_confirmed:
                    # Extreme bullish power - expect reversal
                    new_position = -1
                elif bear_power[i] < -power_threshold_abs and vol_confirmed:
                    # Extreme bearish power - expect reversal
                    new_position = 1
        
        # Execute position change
        if new_position != position:
            signals[i] = new_position * SIGNAL_SIZE
            position = new_position
            entry_price = close[i]
            bars_since_entry = 0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
    
    return signals