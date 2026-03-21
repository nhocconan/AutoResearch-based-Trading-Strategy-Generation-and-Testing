#!/usr/bin/env python3
"""
Experiment #014: 30m HMA Trend + 4h Directional Filter + RSI Pullback
Hypothesis: 30m timeframe captures medium-term swings with less noise than 15m.
4h HMA provides major trend direction (proven in baseline strategies).
RSI pullbacks (40-65 range, not extremes) in trend direction give quality entries.
ATR stoploss at 2.5x protects against 2022-style crashes.
Position size 0.28 (discrete) to limit drawdown while generating sufficient trades.
Key: Relaxed RSI conditions ensure we get 50+ trades/year, not 0 trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_30m_v2"
timeframe = "30m"
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
    """Calculate Hull Moving Average for smooth trend detection."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

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
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)  # auto shift(1)
    
    # Calculate 30m indicators (pre-compute before loop - Rule 8)
    atr = calculate_atr(high, low, close, 14)
    hma_30m = calculate_hma(close, 21)
    rsi = calculate_rsi(close, 14)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=np.mean(volume[:20]) if len(volume) >= 20 else volume[0])
    
    signals = np.zeros(n)
    SIZE = 0.28  # Discrete position size (Rule 4)
    HALF_SIZE = 0.14
    
    # Track positions for stoploss (Rule 6)
    entry_price = 0.0
    position_side = 0  # 0=flat, 1=long, -1=short
    
    for i in range(50, n):
        # 4h trend direction (HTF filter)
        hma_4h_valid = not np.isnan(hma_4h_aligned[i]) and hma_4h_aligned[i] > 0
        trend_4h_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_4h_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # 30m HMA trend (local momentum)
        hma_30m_valid = not np.isnan(hma_30m[i])
        trend_30m_bullish = hma_30m_valid and close[i] > hma_30m[i]
        trend_30m_bearish = hma_30m_valid and close[i] < hma_30m[i]
        
        # RSI pullback conditions (RELAXED to ensure trades - Rule 9)
        # Long: RSI 40-65 (not overbought, but showing momentum)
        rsi_long = rsi[i] >= 40 and rsi[i] <= 68
        # Short: RSI 32-60 (not oversold, but showing weakness)
        rsi_short = rsi[i] >= 32 and rsi[i] <= 60
        
        # Volume confirmation (very relaxed)
        vol_ok = volume[i] > vol_sma[i] * 0.6 if vol_sma[i] > 0 else True
        
        # ATR valid check
        atr_valid = not np.isnan(atr[i]) and atr[i] > 0
        
        new_signal = 0.0
        
        # Long entry: 4h bullish + 30m bullish + RSI ok + volume
        if trend_4h_bullish and trend_30m_bullish and rsi_long and vol_ok:
            new_signal = SIZE
        # Short entry: 4h bearish + 30m bearish + RSI ok
        elif trend_4h_bearish and trend_30m_bearish and rsi_short and vol_ok:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - MUST have this
        if position_side > 0 and entry_price > 0 and atr_valid:
            stop_loss = entry_price - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit - exit long
            # Take partial profit at 3R
            elif close[i] > entry_price + 3.0 * atr[i]:
                if new_signal == SIZE:  # Only reduce if still in full position
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price > 0 and atr_valid:
            stop_loss = entry_price + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit - exit short
            # Take partial profit at 3R
            elif close[i] < entry_price - 3.0 * atr[i]:
                if new_signal == -SIZE:  # Only reduce if still in full position
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            # New position
            entry_price = close[i]
            position_side = np.sign(new_signal)
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                # Position flip
                entry_price = close[i]
                position_side = np.sign(new_signal)
            # Same direction - keep entry price
        elif new_signal == 0 and position_side != 0:
            # Position closed (stoploss or exit signal)
            position_side = 0
            entry_price = 0.0
        
        signals[i] = new_signal
    
    return signals