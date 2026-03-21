#!/usr/bin/env python3
"""
Experiment #007: 15m Hybrid Trend + Mean Reversion + 4h Filter
Hypothesis: 15m timeframe can capture both trend continuations and mean-reversion
pullbacks. 4h HMA provides major trend direction. Entry on RSI pullbacks in trend
direction (buy dips in uptrend, sell rallies in downtrend). Choppiness Index filters
regime to reduce whipsaw. ATR stoploss (2.0x) with partial profit taking.
Relaxed entry thresholds to ensure ≥10 trades/symbol on train data.
Position sizing 0.25-0.35 discrete levels to minimize fee churn while controlling DD.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hybrid_rsi_4h_v1"
timeframe = "15m"
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
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    atr = calculate_atr(high, low, close, period)
    
    choppiness = np.zeros(len(close))
    for i in range(period, len(close)):
        atr_sum = np.sum(atr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            choppiness[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50
    
    choppiness = np.clip(choppiness, 0, 100)
    return choppiness

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))
    
    supertrend[0] = upper[0]
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower[i]
            direction[i] = 1
        else:
            supertrend[i] = upper[i]
            direction[i] = -1
    
    return supertrend, direction

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
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_fast = calculate_rsi(close, 7)
    choppiness = calculate_choppiness(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 2.5)
    
    # EMA for trend confirmation
    ema_fast = pd.Series(close).ewm(span=12, min_periods=12, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=26, min_periods=26, adjust=False).mean().values
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=1.0)
    
    signals = np.zeros(n)
    SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    trailing_stop = np.zeros(n)
    
    for i in range(100, n):
        # 4h trend filter (major regime) - relaxed for more trades
        hma_4h_valid = hma_4h_aligned[i] > 0
        trend_bullish_4h = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_bearish_4h = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # 15m trend signals
        ema_bullish = ema_fast[i] > ema_slow[i]
        ema_bearish = ema_fast[i] < ema_slow[i]
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Choppiness regime
        is_range = choppiness[i] > 50
        is_trend = choppiness[i] < 50
        
        # RSI conditions - relaxed for more trades
        rsi_neutral_long = rsi[i] > 35 and rsi[i] < 60
        rsi_neutral_short = rsi[i] > 40 and rsi[i] < 65
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        
        # Volume confirmation - relaxed
        vol_confirm = volume[i] > vol_sma[i] * 0.7 if vol_sma[i] > 0 else True
        
        # Entry logic - hybrid approach for more trades
        new_signal = 0.0
        
        # Long entry: multiple conditions (any can trigger)
        # 1) 4h bullish + 15m trend + RSI ok
        if trend_bullish_4h and ema_bullish and rsi_neutral_long:
            new_signal = SIZE
        # 2) 4h bullish + Supertrend long + RSI not extreme
        elif trend_bullish_4h and st_long and rsi[i] < 65:
            new_signal = SIZE
        # 3) EMA crossover with 4h support
        elif trend_bullish_4h and ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]:
            new_signal = SIZE
        # 4) Supertrend flip with volume
        elif st_direction[i] == 1 and st_direction[i-1] == -1 and vol_confirm:
            new_signal = SIZE
        # 5) RSI oversold in 4h uptrend (mean reversion)
        elif trend_bullish_4h and rsi_oversold and vol_confirm:
            new_signal = SIZE
        
        # Short entry: multiple conditions (any can trigger)
        # 1) 4h bearish + 15m trend + RSI ok
        elif trend_bearish_4h and ema_bearish and rsi_neutral_short:
            new_signal = -SIZE
        # 2) 4h bearish + Supertrend short + RSI not extreme
        elif trend_bearish_4h and st_short and rsi[i] > 35:
            new_signal = -SIZE
        # 3) EMA crossover with 4h resistance
        elif trend_bearish_4h and ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]:
            new_signal = -SIZE
        # 4) Supertrend flip with volume
        elif st_direction[i] == -1 and st_direction[i-1] == 1 and vol_confirm:
            new_signal = -SIZE
        # 5) RSI overbought in 4h downtrend (mean reversion)
        elif trend_bearish_4h and rsi_overbought and vol_confirm:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.0 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0
            else:
                trailing_stop[i] = max(trailing_stop[i-1] if i > 0 else 0, close[i] - 2.0 * atr[i])
                if close[i] < trailing_stop[i] and trailing_stop[i] > 0:
                    new_signal = 0.0
                elif close[i] > entry_price[i-1] + 2.5 * atr[i] and new_signal == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.0 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0
            else:
                trailing_stop[i] = min(trailing_stop[i-1] if i > 0 else 999999, close[i] + 2.0 * atr[i])
                if close[i] > trailing_stop[i] and trailing_stop[i] < 999999:
                    new_signal = 0.0
                elif close[i] < entry_price[i-1] - 2.5 * atr[i] and new_signal == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price[i] = close[i]
            position_side = np.sign(new_signal)
            trailing_stop[i] = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price[i] = close[i]
                position_side = np.sign(new_signal)
                trailing_stop[i] = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            else:
                trailing_stop[i] = trailing_stop[i-1] if i > 0 else trailing_stop[i]
        else:
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            trailing_stop[i] = trailing_stop[i-1] if i > 0 else 0
            if position_side != 0 and new_signal == 0:
                position_side = 0
        
        signals[i] = new_signal
    
    return signals