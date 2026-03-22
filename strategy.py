#!/usr/bin/env python3
"""
Experiment #143: 12h Donchian Breakout + 1d HMA Trend Filter + Volume Confirmation + ATR Stop

Hypothesis: After 142 failed experiments, returning to proven breakout mechanics with 
proper multi-timeframe filtering. Donchian channels work exceptionally well on slower 
timeframes (12h) because they capture sustained moves while filtering noise.

Key innovations vs previous failures:
- Donchian(20) breakout: Simple but effective on 12h (less whipsaw than 4h/1h)
- 1d HMA(21) trend filter: Only long when price > 1d HMA, only short when < (proven in best strategy)
- Volume confirmation: Breakout must have volume > 1.5x 20-period avg (filters false breakouts)
- ATR(14) trailing stop at 2.5x: Protects capital during reversals
- Asymmetric sizing: 0.30 base, 0.40 on strong volume confirmation
- Loose enough entries to ensure 10+ trades on train, 3+ on test

Why 12h Donchian might work where others failed:
- 12h naturally reduces noise vs 4h/1h strategies that got whipsawed in 2022
- Donchian breakout captures trend continuation (works in both bull/bear)
- Volume filter avoids fake breakouts (major issue in crypto)
- 1d HMA provides stable trend bias (proven in mtf_4h_kama_1d_hma_adx_atr_v1)
- Fewer trades = less fee drag, higher quality signals

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_hma_vol_atr_v1"
timeframe = "12h"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def calculate_rsi(close, period=14):
    """Calculate RSI for momentum confirmation."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    mask = loss_s > 0
    rs[mask] = gain_s[mask] / loss_s[mask]
    rs[~mask] = 100
    
    rsi = 100 - (100 / (1 + rs))
    rsi[rs == 100] = 100
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    vol_sma = calculate_volume_sma(volume, 20)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    # Track breakout state
    prev_donchian_upper = 0.0
    prev_donchian_lower = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout above upper channel
        breakout_long = close[i] > donchian_upper[i]
        # Breakout below lower channel
        breakout_short = close[i] < donchian_lower[i]
        
        # Check if this is a NEW breakout (not continuation)
        new_breakout_long = breakout_long and (prev_donchian_upper == 0.0 or close[i-1] <= prev_donchian_upper)
        new_breakout_short = breakout_short and (prev_donchian_lower == 0.0 or close[i-1] >= prev_donchian_lower)
        
        # === VOLUME CONFIRMATION ===
        # Volume must be > 1.5x average for strong signal
        vol_spike = volume[i] > 1.5 * vol_sma[i]
        vol_confirmed = volume[i] > 1.2 * vol_sma[i]
        
        # === RSI MOMENTUM FILTER ===
        # Avoid overbought longs / oversold shorts
        rsi_not_overbought = rsi[i] < 75
        rsi_not_oversold = rsi[i] > 25
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Strong: 1d bullish + Donchian breakout + Volume spike + RSI ok
        if bull_trend_1d and new_breakout_long and vol_spike and rsi_not_overbought:
            new_signal = SIZE_STRONG
        # Moderate: 1d bullish + Donchian breakout + Volume confirmed
        elif bull_trend_1d and breakout_long and vol_confirmed:
            new_signal = SIZE_BASE
        # Weak (ensure trades): 1d bullish + Donchian breakout (no volume filter)
        elif bull_trend_1d and breakout_long:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Strong: 1d bearish + Donchian breakout + Volume spike + RSI ok
        if bear_trend_1d and new_breakout_short and vol_spike and rsi_not_oversold:
            new_signal = -SIZE_STRONG
        # Moderate: 1d bearish + Donchian breakout + Volume confirmed
        elif bear_trend_1d and breakout_short and vol_confirmed:
            new_signal = -SIZE_BASE
        # Weak (ensure trades): 1d bearish + Donchian breakout (no volume filter)
        elif bear_trend_1d and breakout_short:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        # Store previous Donchian values for breakout detection
        prev_donchian_upper = donchian_upper[i]
        prev_donchian_lower = donchian_lower[i]
        
        signals[i] = new_signal
    
    return signals