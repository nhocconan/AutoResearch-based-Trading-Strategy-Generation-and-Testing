#!/usr/bin/env python3
"""
Experiment #133: 1d Primary + 1w HTF — Donchian Breakout + HMA Trend + RSI Pullback

Hypothesis: After 132 failed experiments, the pattern is clear:
- Complex regime filters (Choppiness, CRSI dual-regime) cause 0 trades or negative Sharpe
- 1d primary timeframe with 1w HTF bias should reduce noise and improve Sharpe
- Donchian breakout (period=20) proven on SOL (+0.782) — simple and effective
- HMA trend filter from 1w provides major direction bias without over-filtering
- RSI pullback confirmation (30/70 thresholds) ensures trades generate on all symbols
- ATR trailing stop (2.5x) for risk management

Key design choices:
- Timeframe: 1d (higher TF = fewer trades, less fee drag, target 20-50 trades/year)
- HTF: 1w for major trend bias (very slow, filters out multi-week noise)
- Entry: Donchian(20) breakout in direction of 1w HMA trend
- Filter: RSI(14) > 30 for longs, < 70 for shorts (loose enough to generate trades)
- Position size: 0.30 (30% of capital, conservative for daily)
- Stoploss: 2.5x ATR trailing (tighter than 3x for better risk control)

Why this should work:
1. 1w HMA is very slow — only changes direction on major trend shifts
2. Donchian breakout catches momentum moves (proven on SOL)
3. RSI filter prevents entering at extreme overbought/oversold
4. 1d timeframe = ~250 bars/year, with filters should get 20-50 trades
5. Simple logic = less likely to have 0 trades from over-filtering

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_rsi_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Helper function for WMA
    def wma(series, span):
        weights = np.arange(1, span + 1)
        weights = weights / weights.sum()
        result = np.convolve(series, weights[::-1], mode='valid')
        return result
    
    close_series = pd.Series(close)
    
    # WMA(n/2)
    wma_half = close_series.ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    # WMA(n)
    wma_full = close_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # 2*WMA(n/2) - WMA(n)
    diff = 2 * wma_half - wma_full
    
    # WMA of diff with sqrt(n) period
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    # Pad with NaN at beginning
    result = np.full(n, np.nan)
    pad_len = n - len(hma)
    if pad_len > 0:
        result[pad_len:] = hma
    else:
        result[:] = hma[-n:]
    
    return result

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian_channels(high, low, period=20):
    """
    Donchian Channels — highest high and lowest low over period
    Upper = rolling max of high
    Lower = rolling min of low
    Middle = (Upper + Lower) / 2
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    
    return upper, lower, middle

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_1d = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian_channels(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (conservative for 1d)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        # Simple: is price above or below weekly HMA?
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d TREND (HMA slope) ===
        # Check if 1d HMA is trending in same direction as HTF
        hma_1d_bull = hma_1d[i] > hma_1d[i-1] if i > 0 else False
        hma_1d_bear = hma_1d[i] < hma_1d[i-1] if i > 0 else False
        
        # === DONCHIAN BREAKOUT ===
        # Long: price breaks above Donchian upper
        # Short: price breaks below Donchian lower
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === RSI FILTER (LOOSE - ensure trades generate on all symbols) ===
        # For longs: RSI > 30 (not extremely oversold)
        # For shorts: RSI < 70 (not extremely overbought)
        rsi_ok_long = rsi[i] > 30.0
        rsi_ok_short = rsi[i] < 70.0
        
        # === DESIRED SIGNAL ===
        # LONG: 1w bull + 1d HMA bull + Donchian breakout + RSI > 30
        # SHORT: 1w bear + 1d HMA bear + Donchian breakout + RSI < 70
        desired_signal = 0.0
        
        if htf_bull and hma_1d_bull and donchian_breakout_long and rsi_ok_long:
            desired_signal = SIZE
        elif htf_bear and hma_1d_bear and donchian_breakout_short and rsi_ok_short:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals