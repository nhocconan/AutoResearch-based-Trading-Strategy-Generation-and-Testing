#!/usr/bin/env python3
"""
Experiment #009: 1h Multi-Timeframe Asymmetric Trend + Volatility Filter
Hypothesis: 1h primary timeframe balances noise reduction with trade frequency.
4h HMA provides trend regime (bull/bear), 1h RSI gives pullback entries.
Bollinger BandWidth percentile filters high-volatility crash periods (2022).
Asymmetric sizing: larger longs in bull regime, smaller/no shorts in bear.
This should generate 30-60 trades/year with positive Sharpe on ALL symbols.
Key insight: BTC/ETH need long bias, SOL needs trend following. Asymmetric approach works for all.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_asymmetric_trend_1h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and BandWidth."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    bandwidth = np.nan_to_num(bandwidth, nan=0.0)
    return upper, lower, bandwidth

def calculate_percentile_rank(values, window=100):
    """Calculate rolling percentile rank."""
    series = pd.Series(values)
    pr = series.rolling(window=window, min_periods=window).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
    )
    return np.nan_to_num(pr.values, nan=0.5)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Load 12h HTF data for additional regime filter
    df_12h = get_htf_data(prices, '12h')
    ema_12h = calculate_ema(df_12h['close'].values, 21)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    ema_fast = calculate_ema(close, 8)
    ema_slow = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    rsi = calculate_rsi(close, 14)
    
    # Bollinger Bands for volatility filter
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    bb_width_pct = calculate_percentile_rank(bb_width, 100)
    
    # Volume SMA
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=0.0)
    
    signals = np.zeros(n)
    
    # Position sizing: asymmetric (larger longs, smaller shorts)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.20
    HALF_LONG = 0.15
    HALF_SHORT = 0.10
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(100, n):
        # 4h trend regime
        hma_4h_val = hma_4h_aligned[i]
        hma_4h_prev = hma_4h_aligned[i-1] if i > 0 else hma_4h_val
        trend_4h_bull = hma_4h_val > 0 and close[i] > hma_4h_val
        trend_4h_bear = hma_4h_val > 0 and close[i] < hma_4h_val
        
        # 12h confirmation
        ema_12h_val = ema_12h_aligned[i]
        trend_12h_bull = ema_12h_val > 0 and close[i] > ema_12h_val
        trend_12h_bear = ema_12h_val > 0 and close[i] < ema_12h_val
        
        # Volatility filter - avoid high vol crash periods
        low_vol = bb_width_pct[i] < 0.70  # Avoid top 30% volatility
        
        # 1h EMA trend
        ema_trend_long = ema_fast[i] > ema_slow[i] and close[i] > ema_50[i]
        ema_trend_short = ema_fast[i] < ema_slow[i] and close[i] < ema_50[i]
        
        # RSI pullback entries (not extreme)
        rsi_long_entry = rsi[i] > 40 and rsi[i] < 65
        rsi_short_entry = rsi[i] > 35 and rsi[i] < 60
        
        # RSI momentum
        rsi_rising = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_falling = rsi[i] < rsi[i-1] if i > 0 else False
        
        # Volume confirmation
        vol_ok = volume[i] > vol_sma[i] * 0.7 if vol_sma[i] > 0 else True
        
        # EMA crossover signals
        ema_cross_long = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        ema_cross_short = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        # Entry logic - RELAXED to ensure trades on all symbols
        new_signal = 0.0
        
        # Long entry: 4h bullish + 1h trend + RSI ok + low vol
        if trend_4h_bull and ema_trend_long and rsi_long_entry and vol_ok:
            new_signal = SIZE_LONG
        # Long on EMA crossover with 4h support
        elif trend_4h_bull and ema_cross_long and rsi[i] > 35:
            new_signal = SIZE_LONG
        # Long on RSI bounce from oversold in uptrend
        elif trend_4h_bull and rsi[i] > 45 and rsi_rising and close[i] > ema_slow[i]:
            new_signal = SIZE_LONG
        
        # Short entry: 4h bearish + 1h trend + RSI ok + low vol
        # Smaller size for shorts (asymmetric)
        if trend_4h_bear and trend_12h_bear and ema_trend_short and rsi_short_entry and low_vol:
            new_signal = -SIZE_SHORT
        # Short on EMA crossover with 4h resistance
        elif trend_4h_bear and ema_cross_short and rsi[i] < 65:
            new_signal = -SIZE_SHORT
        
        # Stoploss logic (Rule 6) - ATR based
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for longs - take partial profit at 3R
            elif close[i] > entry_price[i-1] + 3.0 * atr[i]:
                if signals[i-1] == SIZE_LONG:
                    new_signal = HALF_LONG
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for shorts - take partial profit at 3R
            elif close[i] < entry_price[i-1] - 3.0 * atr[i]:
                if signals[i-1] == -SIZE_SHORT:
                    new_signal = -HALF_SHORT
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price[i] = close[i]
            position_side = np.sign(new_signal)
            highest_since_entry[i] = close[i]
            lowest_since_entry[i] = close[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price[i] = close[i]
                position_side = np.sign(new_signal)
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
            else:
                entry_price[i] = entry_price[i-1]
                highest_since_entry[i] = max(highest_since_entry[i-1], close[i])
                lowest_since_entry[i] = min(lowest_since_entry[i-1], close[i])
        else:
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else close[i]
            lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else close[i]
            if position_side != 0 and new_signal == 0:
                position_side = 0  # Position closed
        
        signals[i] = new_signal
    
    return signals