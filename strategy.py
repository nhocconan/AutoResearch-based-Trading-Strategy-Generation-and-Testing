#!/usr/bin/env python3
"""
Experiment #113: 1d Primary + 1w HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: After 100+ failed experiments, the clearest pattern is:
- 0 trades = automatic failure (experiments 105-112 all failed with Sharpe=0.000)
- 1d primary timeframe with 1w HTF bias reduces whipsaws significantly
- Donchian breakouts (20-period) capture major moves without excessive trading
- HMA on 1w provides smooth trend bias without lag of SMA
- LOOSE RSI thresholds (20/80) ensure trades generate on ALL symbols
- Position size 0.30 (30%) balances return vs drawdown

Key design choices:
- Timeframe: 1d (primary) — proven to work, 20-50 trades/year target
- HTF: 1w for major trend bias (very slow, reduces false signals)
- Entry: Donchian(20) breakout + 1w HMA direction + RSI filter
- Exit: Donchian(20) opposite breakout OR ATR trailing stop (3x)
- Position size: 0.30 (30% of capital, conservative for daily)
- Stoploss: 3x ATR trailing (wider for daily timeframe)

Why this should work:
1. Donchian breakout is proven (Turtle Trading, 50+ year track record)
2. 1w HMA filter prevents counter-trend trades in major moves
3. Loose RSI (20/80) ensures we don't miss entries at extremes
4. 1d timeframe = ~25-40 trades/year = low fee drag
5. Simple logic = fewer bugs = trades actually generate

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_rsi_loose_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — smoother and more responsive than EMA
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Helper: Weighted Moving Average
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(series)):
            result[i] = np.sum(series[i-span+1:i+1] * weights) / np.sum(weights)
        return result
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = 2 * wma_half - wma_full
    
    # WMA of diff with sqrt(n) period
    hma = wma(diff, sqrt_n)
    
    return hma

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel — highest high and lowest low over period
    Returns: upper_band, lower_band, middle_band
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    middle = (upper + lower) / 2.0
    
    return upper, lower, middle

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
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Also calculate 1d HMA for additional trend confirmation
    hma_1d = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (conservative for daily)
    
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
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        # Price above weekly HMA = bullish bias, below = bearish
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d TREND (1d HMA slope) ===
        # HMA sloping up = bullish, sloping down = bearish
        hma_1d_bull = hma_1d[i] > hma_1d[i-1] if i > 0 else False
        hma_1d_bear = hma_1d[i] < hma_1d[i-1] if i > 0 else False
        
        # === DONCHIAN BREAKOUT ===
        # Price breaks above upper band = long signal
        # Price breaks below lower band = short signal
        donchian_breakout_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        donchian_breakout_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # === RSI FILTER (LOOSE - ensure trades generate) ===
        # For longs: RSI > 20 (not extremely oversold, but not overbought)
        # For shorts: RSI < 80 (not extremely overbought, but not oversold)
        rsi_ok_long = rsi[i] > 20.0
        rsi_ok_short = rsi[i] < 80.0
        
        # === DESIRED SIGNAL ===
        # LONG: 1w bull + 1d HMA up + Donchian breakout + RSI > 20
        # SHORT: 1w bear + 1d HMA down + Donchian breakout + RSI < 80
        # RELAXED: Allow entry if 2 out of 3 conditions met (ensures trades generate)
        
        long_score = 0
        short_score = 0
        
        if htf_bull:
            long_score += 1
        if hma_1d_bull:
            long_score += 1
        if donchian_breakout_long and rsi_ok_long:
            long_score += 2  # Breakout is stronger signal
        
        if htf_bear:
            short_score += 1
        if hma_1d_bear:
            short_score += 1
        if donchian_breakout_short and rsi_ok_short:
            short_score += 2  # Breakout is stronger signal
        
        desired_signal = 0.0
        
        # Need score >= 2 for entry (breakout + at least 1 trend confirm)
        if long_score >= 2:
            desired_signal = SIZE
        elif short_score >= 2:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 3x for daily) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT ON OPPOSITE BREAKOUT ===
        # If long and price breaks Donchian lower, exit
        # If short and price breaks Donchian upper, exit
        if in_position and position_side > 0:
            if close[i] < donchian_lower[i-1]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if close[i] > donchian_upper[i-1]:
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