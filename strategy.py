# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels with daily volume confirmation and 1-week trend filter.
# Uses key intraday support/resistance levels (Camarilla) derived from prior day's range,
# filtered by weekly trend to avoid counter-trend trades, and confirmed by volume spikes.
# Works in ranging markets (mean reversion at pivot levels) and trending markets (breakouts).
# Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and cost.

name = "exp_13421_4h_camarilla_pivot_1w_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
CAMARILLA_MULTIPLIER = 1.1  # Standard Camarilla uses 1.1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
WEEKLY_TREND_PERIOD = 50  # EMA for weekly trend filter

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for intraday trading
    Based on previous period's high, low, close
    Returns levels: L4, L3, L2, L1, H1, H2, H3, H4
    """
    range_val = high - low
    close_val = close
    
    # Camarilla levels
    L4 = close_val - (CAMARILLA_MULTIPLIER * range_val / 2)
    L3 = close_val - (CAMARILLA_MULTIPLIER * range_val / 4)
    L2 = close_val - (CAMARILLA_MULTIPLIER * range_val / 6)
    L1 = close_val - (CAMARILLA_MULTIPLIER * range_val / 12)
    H1 = close_val + (CAMARILLA_MULTIPLIER * range_val / 12)
    H2 = close_val + (CAMARILLA_MULTIPLIER * range_val / 6)
    H3 = close_val + (CAMARILLA_MULTIPLIER * range_val / 4)
    H4 = close_val + (CAMARILLA_MULTIPLIER * range_val / 2)
    
    return L4, L3, L2, L1, H1, H2, H3, H4

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_weekly = df_weekly['close'].values
    ema_weekly = calculate_ema(close_weekly, WEEKLY_TREND_PERIOD)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA for confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WEEKLY_TREND_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(ema_weekly_aligned[i]):
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
        
        # Calculate Camarilla levels using previous bar's data
        # Need at least one prior bar
        if i == 0:
            signals[i] = 0.0
            continue
            
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        L4, L3, L2, L1, H1, H2, H3, H4 = calculate_camarilla(prev_high, prev_low, prev_close)
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: weekly EMA direction
        above_weekly_ema = close[i] > ema_weekly_aligned[i]
        below_weekly_ema = close[i] < ema_weekly_aligned[i]
        
        # Trading logic:
        # In uptrend: look for longs at support levels (L3, L4) and avoid shorts
        # In downtrend: look for shorts at resistance levels (H3, H4) and avoid longs
        # In ranging: trade both sides with tighter stops
        
        if position == 0:
            # Long conditions: price at support with volume and trend alignment
            long_signal = False
            if above_weekly_ema:  # Uptrend - buy dips to support
                if (low[i] <= L3 and close[i] > L3) or (low[i] <= L4 and close[i] > L4):
                    long_signal = volume_ok
            elif below_weekly_ema:  # Downtrend - only trade strong bounces
                if low[i] <= L4 and close[i] > L4:
                    long_signal = volume_ok and (close[i] > (L4 + L3) / 2)  # Need stronger bounce
            else:  # Ranging - trade both sides
                if low[i] <= L3 and close[i] > L3:
                    long_signal = volume_ok
            
            # Short conditions: price at resistance with volume and trend alignment
            short_signal = False
            if below_weekly_ema:  # Downtrend - sell rallies to resistance
                if (high[i] >= H3 and close[i] < H3) or (high[i] >= H4 and close[i] < H4):
                    short_signal = volume_ok
            elif above_weekly_ema:  # Uptrend - only trade strong pullbacks
                if high[i] >= H4 and close[i] < H4:
                    short_signal = volume_ok and (close[i] < (H4 + H3) / 2)  # Need stronger pullback
            else:  # Ranging - trade both sides
                if high[i] >= H3 and close[i] < H3:
                    short_signal = volume_ok
            
            # Generate signals
            if long_signal and not short_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal and not long_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Manage long position: exit if price reaches resistance or stops
            if close[i] >= H3:  # Take profit at resistance
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Manage short position: exit if price reaches support or stops
            if close[i] <= L3:  # Take profit at support
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals