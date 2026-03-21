#!/usr/bin/env python3
"""
Experiment #050: 30m Donchian Breakout with 4h+1d Dual HMA Trend Filter + RSI Pullback
Hypothesis: 30m needs stronger HTF confirmation than 12h to avoid whipsaw. Use BOTH
4h and 1d HMA aligned in same direction (double trend filter). Entry via Donchian(20)
breakout for clean momentum signals, but ONLY when RSI confirms (not overextended).
RSI pullback entries when trend is strong but price dips (RSI 40-50 in uptrend).
This combines the proven Supertrend logic from #047 with Donchian breakouts which
work better on 30m than pure mean-reversion. Wider 2.5*ATR stop to reduce 30m noise.
Position sizing: 0.25 entry, 0.125 at 2R profit, max 0.30 on strong confirmations.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_donchian_4h_1d_hma_rsi_v1"
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
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = period // 2
    if half < 1:
        half = 1
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, lower, middle

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    sma_200 = calculate_sma(close, 200)
    
    # 30m HMA for local trend
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    SIZE_STRONG = 0.30
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(donchian_upper[i]) or np.isnan(sma_200[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Dual HTF trend filter - BOTH 4h and 1d must agree
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        hma_1d_bullish = close[i] > hma_1d_aligned[i]
        hma_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # Strong trend: both 4h and 1d agree
        strong_uptrend = hma_4h_bullish and hma_1d_bullish
        strong_downtrend = hma_4h_bearish and hma_1d_bearish
        
        # Also check SMA200 for additional confirmation
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # 30m local trend
        hma_trend_long = hma_21[i] > hma_50[i]
        hma_trend_short = hma_21[i] < hma_50[i]
        
        # Donchian breakout signals
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # Volume confirmation (above average)
        volume_confirmed = volume[i] > vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # RSI conditions
        rsi_neutral_long = 35 < rsi[i] < 55  # Not overbought, room to run
        rsi_neutral_short = 45 < rsi[i] < 65  # Not oversold, room to fall
        rsi_pullback_long = 40 < rsi[i] < 50  # Dip in uptrend
        rsi_pullback_short = 50 < rsi[i] < 60  # Rally in downtrend
        rsi_rising = (i > 2) and (rsi[i] > rsi[i-2])
        rsi_falling = (i > 2) and (rsi[i] < rsi[i-2])
        
        new_signal = 0.0
        
        # LONG ENTRY conditions
        # 1. Donchian breakout with strong trend + volume
        if donchian_breakout_long and strong_uptrend and volume_confirmed:
            new_signal = SIZE_STRONG
        # 2. Donchian breakout with 4h trend + HMA confirmation
        elif donchian_breakout_long and hma_4h_bullish and hma_trend_long:
            new_signal = SIZE_ENTRY
        # 3. RSI pullback in strong uptrend (buy the dip)
        elif strong_uptrend and rsi_pullback_long and rsi_rising and hma_trend_long:
            new_signal = SIZE_ENTRY
        # 4. HMA crossover in strong trend
        elif strong_uptrend and hma_trend_long and above_sma200:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY conditions
        # 1. Donchian breakdown with strong trend + volume
        if donchian_breakout_short and strong_downtrend and volume_confirmed:
            new_signal = -SIZE_STRONG
        # 2. Donchian breakdown with 4h trend + HMA confirmation
        elif donchian_breakout_short and hma_4h_bearish and hma_trend_short:
            new_signal = -SIZE_ENTRY
        # 3. RSI pullback in strong downtrend (sell the rally)
        elif strong_downtrend and rsi_pullback_short and rsi_falling and hma_trend_short:
            new_signal = -SIZE_ENTRY
        # 4. HMA crossover in strong trend
        elif strong_downtrend and hma_trend_short and below_sma200:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Calculate trailing stop (2.5*ATR for 30m noise)
            current_stop = close[i] - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced:
                    profit = close[i] - entry_price
                    risk = 2.5 * atr[int(i)] if i > 0 else atr[i]
                    if risk > 0 and profit >= 2.0 * risk:
                        new_signal = SIZE_HALF
                        position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Calculate trailing stop
            current_stop = close[i] + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced:
                    profit = entry_price - close[i]
                    risk = 2.5 * atr[int(i)] if i > 0 else atr[i]
                    if risk > 0 and profit >= 2.0 * risk:
                        new_signal = -SIZE_HALF
                        position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals