#!/usr/bin/env python3
"""
Experiment #015: 1h Donchian Breakout + 4h HMA Trend Filter + Volume
Hypothesis: 1h timeframe captures medium-term swings better than intraday noise.
4h HMA provides major trend regime filter (only trade in trend direction).
Donchian(20) breakout gives clear entry signals with volume confirmation.
ATR-based stoploss (2.5x) protects against crashes like 2022.
Position sizing capped at 0.25 with discrete levels to minimize fee churn.
This combines successful elements from mtf_4h_donchian_daily_v1 (current best)
but uses 1h entries for more responsive signals while keeping 4h trend filter.
Relaxed entry conditions to ensure ≥10 trades/symbol on train data.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_donchian_4h_hma_vol_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    rsi = calculate_rsi(close, 14)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=np.nanmean(volume))
    
    # Price momentum (ROC)
    roc = np.zeros(n)
    roc[10:] = (close[10:] - close[:-10]) / close[:-10] * 100
    
    signals = np.zeros(n)
    SIZE = 0.25
    HALF_SIZE = 0.12
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    trailing_stop = np.zeros(n)
    max_close = np.zeros(n)
    min_close = np.zeros(n)
    
    for i in range(100, n):
        # 4h trend filter (major regime)
        hma_4h_valid = hma_4h_aligned[i] > 0
        trend_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # Donchian breakout signals
        donchian_breakout_long = close[i] > donchian_upper[i-1] if donchian_upper[i-1] > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if donchian_lower[i-1] > 0 else False
        
        # Previous bar check (ensure breakout is fresh)
        prev_close = close[i-1]
        prev_upper = donchian_upper[i-2] if i > 1 else donchian_upper[i-1]
        prev_lower = donchian_lower[i-2] if i > 1 else donchian_lower[i-1]
        
        breakout_confirmed_long = prev_close <= prev_upper and close[i] > donchian_upper[i-1]
        breakout_confirmed_short = prev_close >= prev_lower and close[i] < donchian_lower[i-1]
        
        # Volume confirmation (relaxed - 80% of average is ok)
        vol_confirm = volume[i] > vol_sma[i] * 0.8 if vol_sma[i] > 0 else True
        
        # RSI filter (avoid extreme overbought/oversold on entry)
        rsi_ok_long = rsi[i] < 75  # Not extremely overbought
        rsi_ok_short = rsi[i] > 25  # Not extremely oversold
        
        # Momentum confirmation
        momentum_long = roc[i] > 0
        momentum_short = roc[i] < 0
        
        # Entry logic - relaxed conditions to ensure trades
        new_signal = 0.0
        
        # Long entry: 4h bullish + Donchian breakout + volume + RSI ok
        if trend_bullish and breakout_confirmed_long and vol_confirm and rsi_ok_long:
            new_signal = SIZE
        # Long on momentum + trend alignment (backup entry)
        elif trend_bullish and momentum_long and rsi[i] > 45 and rsi[i] < 70 and vol_confirm:
            new_signal = SIZE
        # Long on RSI pullback in uptrend
        elif trend_bullish and rsi[i] < 40 and rsi[i-3] < rsi[i] and vol_confirm:
            new_signal = SIZE
        
        # Short entry: 4h bearish + Donchian breakout + volume + RSI ok
        elif trend_bearish and breakout_confirmed_short and vol_confirm and rsi_ok_short:
            new_signal = -SIZE
        # Short on momentum + trend alignment (backup entry)
        elif trend_bearish and momentum_short and rsi[i] > 30 and rsi[i] < 55 and vol_confirm:
            new_signal = -SIZE
        # Short on RSI pullback in downtrend
        elif trend_bearish and rsi[i] > 60 and rsi[i-3] > rsi[i] and vol_confirm:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for longs
                current_trail = close[i] - 2.5 * atr[i]
                trailing_stop[i] = max(trailing_stop[i-1] if i > 0 else 0, current_trail)
                if close[i] < trailing_stop[i] and trailing_stop[i] > 0:
                    new_signal = 0.0
                # Take partial profit at 3R
                elif close[i] > entry_price[i-1] + 3.0 * atr[i] and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for shorts
                current_trail = close[i] + 2.5 * atr[i]
                if trailing_stop[i-1] == 0:
                    trailing_stop[i] = current_trail
                else:
                    trailing_stop[i] = min(trailing_stop[i-1], current_trail)
                if close[i] > trailing_stop[i] and trailing_stop[i] < 999999:
                    new_signal = 0.0
                # Take partial profit at 3R
                elif close[i] < entry_price[i-1] - 3.0 * atr[i] and signals[i-1] == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price[i] = close[i]
            position_side = np.sign(new_signal)
            trailing_stop[i] = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price[i] = close[i]
                position_side = np.sign(new_signal)
                trailing_stop[i] = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            else:
                trailing_stop[i] = trailing_stop[i-1] if i > 0 else trailing_stop[i]
        else:
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            trailing_stop[i] = trailing_stop[i-1] if i > 0 else 0
            if position_side != 0 and new_signal == 0:
                position_side = 0  # Position closed
        
        signals[i] = new_signal
    
    return signals