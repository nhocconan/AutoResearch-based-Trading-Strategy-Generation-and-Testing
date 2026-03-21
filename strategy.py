#!/usr/bin/env python3
"""
Hypothesis: KAMA adaptive trend + Donchian breakout on 1d with 1w HTF trend filter.
KAMA adapts to volatility (slow in chop, fast in trends). Donchian captures breakouts.
Weekly HMA filter avoids counter-trend trades. Volume confirmation reduces false signals.
ATR stoploss controls drawdown. Designed for sufficient trade frequency on daily data.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_donchian_1d_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        # Efficiency Ratio
        if i >= er_period:
            signal = abs(close[i] - close[i - er_period])
            noise = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
            er = signal / noise if noise > 0 else 0
        else:
            er = 1.0
        
        # Smoothing constant
        sc = (er * (2.0/(fast_period+1) - 2.0/(slow_period+1)) + 2.0/(slow_period+1)) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load weekly HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly HMA for trend filter
    close_1w = df_1w['close'].values
    hma_1w = calculate_hma(close_1w, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Daily indicators
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    vol_s = pd.Series(volume)
    
    # KAMA adaptive trend
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_slope = np.diff(kama, prepend=kama[0])
    
    # Donchian channels (10-day for more signals on daily)
    donchian_upper = high_s.rolling(window=10, min_periods=10).max().values
    donchian_lower = low_s.rolling(window=10, min_periods=10).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Volume SMA
    vol_sma = vol_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR for stops
    atr = calculate_atr(high, low, close, 14)
    
    # ROC for momentum confirmation
    roc = (close - np.roll(close, 10)) / np.roll(close, 10) * 100
    roc[:10] = 0
    
    signals = np.zeros(n)
    SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track positions for stoploss/takeprofit
    entry_price = 0.0
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(30, n):
        # Weekly trend filter (use previous week's value via alignment)
        weekly_trend = 1 if hma_1w_aligned[i] > hma_1w_aligned[i-1] else -1 if hma_1w_aligned[i] < hma_1w_aligned[i-1] else 0
        
        # KAMA trend direction
        kama_trend = 1 if kama_slope[i] > 0 else -1
        
        # Donchian breakout (compare to previous bar's levels to avoid look-ahead)
        breakout_long = close[i] > donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1]
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.2 * vol_sma[i]
        
        # Momentum confirmation
        momentum_ok_long = roc[i] > 0
        momentum_ok_short = roc[i] < 0
        
        # Entry logic - need weekly trend + KAMA + breakout + volume + momentum
        if position_side == 0:
            # Long entry
            if breakout_long and weekly_trend >= 0 and kama_trend >= 0 and volume_confirmed and momentum_ok_long:
                signals[i] = SIZE
                entry_price = close[i]
                position_side = 1
                highest_since_entry = close[i]
            # Short entry
            elif breakout_short and weekly_trend <= 0 and kama_trend <= 0 and volume_confirmed and momentum_ok_short:
                signals[i] = -SIZE
                entry_price = close[i]
                position_side = -1
                lowest_since_entry = close[i]
        else:
            if position_side == 1:
                # Update highest since entry
                highest_since_entry = max(highest_since_entry, close[i])
                
                # Stoploss: 2.5*ATR from entry
                stoploss_price = entry_price - 2.5 * atr[i]
                
                # Trailing stop after 1.5R profit
                profit_r = (highest_since_entry - entry_price) / atr[i] if atr[i] > 0 else 0
                trail_stop = highest_since_entry - 1.5 * atr[i]
                
                if close[i] < stoploss_price or (profit_r >= 1.5 and close[i] < trail_stop):
                    signals[i] = 0.0
                    position_side = 0
                elif profit_r >= 2.0:
                    # Take partial profit at 2R
                    signals[i] = HALF_SIZE
                else:
                    signals[i] = SIZE
            else:
                # Short position
                lowest_since_entry = min(lowest_since_entry, close[i])
                
                stoploss_price = entry_price + 2.5 * atr[i]
                profit_r = (entry_price - lowest_since_entry) / atr[i] if atr[i] > 0 else 0
                trail_stop = lowest_since_entry + 1.5 * atr[i]
                
                if close[i] > stoploss_price or (profit_r >= 1.5 and close[i] > trail_stop):
                    signals[i] = 0.0
                    position_side = 0
                elif profit_r >= 2.0:
                    signals[i] = -HALF_SIZE
                else:
                    signals[i] = -SIZE
    
    return signals

def calculate_hma(prices, period):
    """Hull Moving Average for smooth trend detection"""
    prices_s = pd.Series(prices)
    wma1 = prices_s.rolling(window=period//2, min_periods=period//2).mean()
    wma2 = prices_s.rolling(window=period, min_periods=period).mean()
    hma_raw = 2 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = hma_raw.rolling(window=sqrt_period, min_periods=sqrt_period).mean().values
    return hma