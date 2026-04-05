#!/usr/bin/env python3
"""
Experiment #8554: 1h momentum + 4h/1d trend filter + volume confirmation + session filter.
Hypothesis: 1h timeframe balances trade frequency and execution speed. Using 4h trend (EMA50) and 1d trend (EMA200) as filters ensures alignment with higher timeframe momentum, reducing counter-trend trades. Volume confirmation ensures institutional participation. Session filter (08-20 UTC) avoids low-volume Asian session noise. Target 60-150 total trades over 4 years (15-37/year) to minimize fee impact.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8554_1h_momentum_4h_1d_trend_vol_session_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_FAST = 12
EMA_SLOW = 26
TREND_4H_PERIOD = 50
TREND_1D_PERIOD = 200
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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
    ema_4h = pd.Series(close_4h).ewm(span=TREND_4H_PERIOD, adjust=False, min_periods=TREND_4H_PERIOD).mean().values
    # Price relative to 4h EMA: above = bullish bias, below = bearish bias
    price_vs_ema_4h = np.where(close_4h > ema_4h, 1, 
                       np.where(close_4h < ema_4h, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_4h_aligned = align_htf_to_ltf(prices, df_4h, price_vs_ema_4h)
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=TREND_1D_PERIOD, adjust=False, min_periods=TREND_1D_PERIOD).mean().values
    price_vs_ema_1d = np.where(close_1d > ema_1d, 1, 
                       np.where(close_1d < ema_1d, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_1d_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema_1d)
    
    # Calculate LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA crossover
    ema_fast = pd.Series(close).ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean().values
    ema_slow = pd.Series(close).ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # Already datetime64[ms], .hour works
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_SLOW, TREND_4H_PERIOD, TREND_1D_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available or outside session
        if np.isnan(price_vs_ema_4h_aligned[i]) or np.isnan(price_vs_ema_1d_aligned[i]) or not in_session[i]:
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
        
        # Determine market bias from HTF EMAs
        bull_bias = (price_vs_ema_4h_aligned[i] == 1) and (price_vs_ema_1d_aligned[i] == 1)
        bear_bias = (price_vs_ema_4h_aligned[i] == -1) and (price_vs_ema_1d_aligned[i] == -1)
        
        # EMA crossover conditions
        ema_cross_up = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        ema_cross_down = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bull_bias and ema_cross_up and volume_confirmed
        short_entry = bear_bias and ema_cross_down and volume_confirmed
        
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
</p>