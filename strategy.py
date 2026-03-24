#!/usr/bin/env python3
"""
Experiment #117: 1d Primary + 1w HTF — Donchian Breakout + HMA Trend + Loose RSI

Hypothesis: Based on exp#113 (mtf_1d_donchian_hma_rsi_loose_1w_v1) which achieved 
Sharpe=0.233 with 1d+1w combination, I will refine this approach:

1. 1w HMA = major trend bias (price above/below weekly HMA)
2. 1d Donchian(20) breakout = entry trigger (price breaks 20-day high/low)
3. RSI(14) loose filter (>20 for long, <80 for short) - ensures trades generate
4. ATR trailing stoploss (3.0x) for risk management
5. Volume confirmation: breakout volume > 1.2x 20-day avg volume

Why this should work better than exp#113:
- Added volume filter to confirm breakouts (reduces false signals)
- HMA instead of SMA for trend (more responsive, less lag)
- Slightly looser RSI (20/80 vs 25/75) to ensure trade generation
- 1d timeframe = 10-30 trades/year target (matches cost model)
- Position size: 0.30 (30% of capital, conservative for daily)

Key design choices:
- Timeframe: 1d (proven in exp#113, low fee drag)
- HTF: 1w for trend bias (major trend direction)
- Donchian(20): classic breakout, works well on daily
- Volume filter: confirms genuine breakouts vs fakeouts
- Stoploss: 3.0x ATR trailing (wider for daily timeframe)

Target: Sharpe>0.351 (beat current best), DD>-40%, trades>=10 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_vol_rsi_1w_v2"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average (HMA)
    Reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Helper function for WMA
    def wma(data, span):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(data)):
            result[i] = np.sum(data[i - span + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    half_period = period // 2
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    # HMA calculation
    hma = np.full(n, np.nan)
    for i in range(period, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            # Apply WMA to the difference with sqrt(period)
            sqrt_period = int(np.sqrt(period))
            if i >= sqrt_period - 1:
                hma[i] = wma(np.concatenate([[diff] * (sqrt_period - 1), [diff]]), sqrt_period)[-1]
            else:
                hma[i] = diff
    
    return hma

def calculate_hma_simple(close, period=21):
    """
    Simplified HMA using pandas for better performance
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    
    # WMA using ewm with adjust=False approximates WMA behavior
    wma_half = close_s.ewm(span=half_period, min_periods=half_period, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # HMA formula
    hma_raw = 2.0 * wma_half - wma_full
    hma = hma_raw.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    
    return hma.values

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel
    Upper = highest high over period
    Lower = lowest low over period
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

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

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma_simple(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_1d = calculate_hma_simple(close, period=21)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
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
        if np.isnan(hma_1d[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_1w_aligned[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        # Simple: is price above or below weekly HMA?
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d TREND (HMA direction) ===
        # Price above 1d HMA = bull, below = bear
        trend_bull = close[i] > hma_1d[i]
        trend_bear = close[i] < hma_1d[i]
        
        # === DONCHIAN BREAKOUT ===
        # Long: price breaks above 20-day high
        # Short: price breaks below 20-day low
        breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # === VOLUME CONFIRMATION ===
        # Breakout volume > 1.2x 20-day average
        vol_confirmed = volume[i] > 1.2 * vol_sma[i] if vol_sma[i] > 1e-10 else False
        
        # === RSI FILTER (VERY LOOSE - ensure trades generate on all symbols) ===
        # For longs: RSI > 20 (not extremely oversold)
        # For shorts: RSI < 80 (not extremely overbought)
        rsi_ok_long = rsi[i] > 20.0
        rsi_ok_short = rsi[i] < 80.0
        
        # === DESIRED SIGNAL ===
        # LONG: 1w bull + 1d trend bull + Donchian breakout + volume + RSI > 20
        # SHORT: 1w bear + 1d trend bear + Donchian breakout + volume + RSI < 80
        desired_signal = 0.0
        
        if htf_bull and trend_bull and breakout_long and vol_confirmed and rsi_ok_long:
            desired_signal = SIZE
        elif htf_bear and trend_bear and breakout_short and vol_confirmed and rsi_ok_short:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 3.0x) ===
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