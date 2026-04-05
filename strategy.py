#!/usr/bin/env python3
"""
Experiment #9174: 1h Donchian breakout + 4h/1d trend filter + volume confirmation + session filter.
Hypothesis: 1h timeframe captures intraday momentum; 4h/1d trend filter ensures directional alignment with higher timeframe trends; volume confirmation filters false breakouts; session filter (08-20 UTC) reduces noise during low-liquidity periods. Targets 60-150 total trades over 4 years (15-37/year) to balance opportunity and cost. Works in bull (breakouts) and bear (filtered shorts).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_9174_1h_donchian20_4h_1d_trend_vol_sess_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
TREND_PERIOD_4H = 30
TREND_PERIOD_1D = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.2

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=TREND_PERIOD_4H, adjust=False, min_periods=TREND_PERIOD_4H).mean().values
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=TREND_PERIOD_1D, adjust=False, min_periods=TREND_PERIOD_1D).mean().values
    
    # Price relative to 4h EMA: above = bullish bias, below = bearish bias
    price_vs_ema_4h = np.where(close_4h > ema_4h, 1, 
                       np.where(close_4h < ema_4h, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_4h_aligned = align_htf_to_ltf(prices, df_4h, price_vs_ema_4h)
    
    # Price relative to 1d EMA: above = bullish bias, below = bearish bias
    price_vs_ema_1d = np.where(close_1d > ema_1d, 1, 
                       np.where(close_1d < ema_1d, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_1d_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_PERIOD_4H, TREND_PERIOD_1D, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_4h_aligned[i]) or np.isnan(price_vs_ema_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check session filter
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            stop_price = 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                stop_price = 0.0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                stop_price = 0.0
                continue
        
        # Determine market bias from 4h/1d EMA (both must agree)
        bull_bias = (price_vs_ema_4h_aligned[i] == 1) and (price_vs_ema_1d_aligned[i] == 1)
        bear_bias = (price_vs_ema_4h_aligned[i] == -1) and (price_vs_ema_1d_aligned[i] == -1)
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
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
</|tool_call|>