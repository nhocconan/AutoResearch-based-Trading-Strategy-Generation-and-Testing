# 1d HTF trend with volume confirmation and volume breakout
# Timeframe: 1d, HTF: 1w (weekly trend)
# Hypothesis: Weekly trend filters daily breakouts to avoid counter-trend trades, volume confirms breakout strength, and volume spikes add momentum confirmation. Works in bull/bear by following higher timeframe trend.
# Target: 15-25 trades/year (60-100 total over 4 years) to stay within optimal range.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12664_1d_1w_trend_vol_breakout_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
BREAKOUT_LOOKBACK = 20          # Donchian breakout period
VOLUME_MA_PERIOD = 20           # Volume moving average
VOLUME_BREAKOUT_MULT = 2.0      # Volume must be 2x average
VOLUME_CONFIRM_MULT = 1.5       # Volume confirmation (1.5x average)
WEEKLY_EMA_PERIOD = 21          # Weekly trend EMA
ATR_PERIOD = 14                 # ATR for stop loss
ATR_STOP_MULT = 2.5             # ATR multiplier for stop loss
SIGNAL_SIZE = 0.25              # Position size (25% of capital)

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_donchian(high, low, period):
    """Calculate Donchian channels with proper min_periods"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    ema_1w = calculate_ema(df_1w['close'].values, WEEKLY_EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    upper, lower = calculate_donchian(high, low, BREAKOUT_LOOKBACK)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(BREAKOUT_LOOKBACK, WEEKLY_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(ema_1w_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
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
        
        # Volume filters
        vol_ma = volume_ma[i]
        vol_breakout = volume[i] > (vol_ma * VOLUME_BREAKOUT_MULT) if not np.isnan(vol_ma) else False
        vol_confirm = volume[i] > (vol_ma * VOLUME_CONFIRM_MULT) if not np.isnan(vol_ma) else False
        
        # Weekly trend filter
        uptrend_1w = close[i] > ema_1w_aligned[i]
        downtrend_1w = close[i] < ema_1w_aligned[i]
        
        # Donchian breakout conditions (using previous bar's levels)
        long_breakout = close[i] > upper[i-1]
        short_breakout = close[i] < lower[i-1]
        
        # Entry conditions: need volume breakout OR confirmation + weekly trend alignment
        long_entry = (vol_breakout or vol_confirm) and uptrend_1w and long_breakout
        short_entry = (vol_breakout or vol_confirm) and downtrend_1w and short_breakout
        
        # Generate signals
        if position == 0:
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