#!/usr/bin/env python3
"""
Experiment #8474: 1h Donchian breakout + 4h/1d trend filter + volume confirmation + session filter.
Hypothesis: On 1h timeframe, use 4h and 1d timeframes for trend direction (reducing counter-trend trades),
1h for precise entry timing via Donchian breakouts, volume confirmation to ensure institutional participation,
and session filter (08-20 UTC) to avoid low-liquidity hours. Target 60-150 total trades over 4 years
(15-37/year) to balance statistical validity with fee minimization. Trend filter avoids counter-trend trades
in both bull and bear markets by aligning with higher timeframe momentum.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8474_1h_donchian20_4h1d_trend_vol_sess_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
TREND_PERIOD_4H = 20
TREND_PERIOD_1D = 50
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
    ema_4h = pd.Series(close_4h).ewm(span=TREND_PERIOD_4H, adjust=False, min_periods=TREND_PERIOD_4H).mean().values
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=TREND_PERIOD_1D, adjust=False, min_periods=TREND_PERIOD_1D).mean().values
    
    # Price relative to EMAs: above = bullish bias, below = bearish bias
    trend_4h = np.where(close_4h > ema_4h, 1, 
                 np.where(close_4h < ema_4h, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    trend_1d = np.where(close_1d > ema_1d, 1, 
                 np.where(close_1d < ema_1d, -1, 0))
    
    # Align HTF trends to 1h timeframe
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Combined trend: require both 4h and 1d to agree
    combined_trend = np.where((trend_4h_aligned == 1) & (trend_1d_aligned == 1), 1,  # bullish
                     np.where((trend_4h_aligned == -1) & (trend_1d_aligned == -1), -1, 0))  # bearish
    
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
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_PERIOD_4H, TREND_PERIOD_1D, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(combined_trend[i]):
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
        
        # Session filter: only trade during 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: maintain current position or flat
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Determine market bias from combined trend
        bull_bias = combined_trend[i] == 1   # Both 4h and 1d bullish
        bear_bias = combined_trend[i] == -1  # Both 4h and 1d bearish
        
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