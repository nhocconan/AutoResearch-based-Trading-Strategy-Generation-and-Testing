#!/usr/bin/env python3
"""
exp_7119_6h_cci_12h_regime_v1
Hypothesis: 6h CCI(20) mean reversion with 12h ADX regime filter.
In low volatility ranging markets (12h ADX < 20): fade CCI extremes (>100 or <-100) with volume confirmation.
In trending markets (12h ADX > 25): avoid mean reversion trades to prevent whipsaw.
Uses 12h ADX for regime detection and 6h CCI for timing, targeting 12-37 trades/year.
Designed to work in both bull and bear markets by adapting to volatility regime.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7119_6h_cci_12h_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CCI_PERIOD = 20
CCI_OVERBOUGHT = 100
CCI_OVERSOLD = -100
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
ADX_PERIOD = 14
ADX_RANGE_THRESHOLD = 20  # Below this = ranging (mean revert)
ADX_TREND_THRESHOLD = 25   # Above this = trending (avoid mean reversion)
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 12  # ~12 * 6h = 3 days

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for ADX regime
    df_12h = get_htf_data(prices, '12h')
    
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
    atr_12h = tr_12h.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    plus_di_12h = 100 * (pd.Series(plus_dm).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values / atr_12h)
    minus_di_12h = 100 * (pd.Series(minus_dm).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values / atr_12h)
    
    # DX and ADX
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h + 1e-10)
    adx_12h = pd.Series(dx_12h).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # Align ADX to LTF (6h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # CCI calculation
    tp = (high + low + close) / 3  # Typical Price
    ma_tp = pd.Series(tp).rolling(window=CCI_PERIOD, min_periods=CCI_PERIOD).mean().values
    mad = pd.Series(tp).rolling(window=CCI_PERIOD, min_periods=CCI_PERIOD).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci = (tp - ma_tp) / (0.015 * mad)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(CCI_PERIOD, VOL_MA_PERIOD, ATR_PERIOD, ADX_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(adx_12h_aligned[i]):
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
        
        # Regime detection: ranging vs trending
        is_ranging = adx_12h_aligned[i] < ADX_RANGE_THRESHOLD
        is_trending = adx_12h_aligned[i] > ADX_TREND_THRESHOLD
        
        # Mean reversion signals only in ranging markets
        cci_long_signal = (cci[i] <= CCI_OVERSOLD) and vol_confirmed
        cci_short_signal = (cci[i] >= CCI_OVERBOUGHT) and vol_confirmed
        
        # Enter new positions only if flat and in ranging regime
        if position == 0 and is_ranging:
            if cci_long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif cci_short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position or stay flat in trending markets
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
    
    return signals