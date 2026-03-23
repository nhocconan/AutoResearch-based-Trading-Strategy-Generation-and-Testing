#!/usr/bin/env python3
"""
Experiment #1176: 12h Primary + 1d HTF — HMA Trend + Donchian Breakout + Volume + RSI

Hypothesis: Building on #1174's success (Sharpe=0.194, positive returns), adapt proven
logic for 12h timeframe. Key insights from 859+ failed experiments:
- 12h naturally produces fewer trades (20-50/year target) → less fee drag
- 1d HTF provides macro trend filter without being too slow
- Volume confirmation reduces false breakouts (major crypto issue)
- RSI momentum filter (not extreme) ensures trend has strength
- Simpler entry conditions than failed regime strategies (#1166, #1172 got 0 trades)
- ATR 3.0x trailing stop appropriate for 12h volatility (wider than 4h)

Why this should beat Sharpe=0.612:
- 12h timeframe = fewer whipsaws than 4h during 2022 crash
- 1d HMA filter prevents counter-trend trades in major moves
- Volume + RSI confluence filters fakeouts while still generating trades
- Discrete signal sizes (0.0, ±0.30) minimize fee churn
- Target: 25-45 trades/year on 12h, Sharpe > 0.612

Timeframe: 12h (primary)
HTF: 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.30 base (discrete: 0.0, ±0.30)
Stoploss: 3.0x ATR trailing (wider for 12h volatility)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_donchian_volume_rsi_1d_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """
    Relative Strength Index — momentum oscillator.
    RSI > 52 = bullish momentum, RSI < 48 = bearish momentum
    """
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel — breakout indicator.
    Upper = highest high over period
    Lower = lowest low over period
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_volume_spike(volume, period=20, threshold=1.3):
    """
    Volume spike detection — confirms breakout validity.
    Lower threshold (1.3x) for 12h to ensure trades trigger
    """
    n = len(volume)
    spike = np.zeros(n, dtype=bool)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    
    for i in range(period - 1, n):
        if vol_avg[i] > 1e-10 and volume[i] > threshold * vol_avg[i]:
            spike[i] = True
    
    return spike

def calculate_sma(close, period=200):
    """Simple Moving Average for trend filter."""
    n = len(close)
    sma = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi_12h = calculate_rsi(close, period=14)
    hma_12h = calculate_hma(close, period=21)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    volume_spike = calculate_volume_spike(volume, period=20, threshold=1.3)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after 200 SMA is ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi_12h[i]) or np.isnan(hma_12h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === LOCAL TREND (12h HMA) ===
        local_bull = close[i] > hma_12h[i]
        local_bear = close[i] < hma_12h[i]
        
        # === LONG-TERM TREND (SMA 200) ===
        long_bull = close[i] > sma_200[i]
        long_bear = close[i] < sma_200[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1]
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume_spike[i]
        
        # === RSI MOMENTUM FILTER (relaxed for 12h) ===
        rsi_bullish = rsi_12h[i] > 52.0
        rsi_bearish = rsi_12h[i] < 48.0
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Macro bull OR long bull + local bull + Donchian breakout + (volume OR RSI)
        # Relaxed logic to ensure trades trigger (learned from 0-trade failures)
        long_trend = (macro_bull or long_bull) and local_bull
        long_momentum = breakout_long and (vol_confirmed or rsi_bullish)
        
        if long_trend and long_momentum:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY ===
        bear_trend = (macro_bear or long_bear) and local_bear
        bear_momentum = breakout_short and (vol_confirmed or rsi_bearish)
        
        if bear_trend and bear_momentum:
            desired_signal = -BASE_SIZE
        
        # === MACRO TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and macro_bear:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bull:
            desired_signal = 0.0
        
        # === LOCAL TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and local_bear:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and local_bull:
            desired_signal = 0.0
        
        # === STOPLOSS CHECK (Trailing ATR 3.0x for 12h) ===
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                if (macro_bull or long_bull) and local_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                if (macro_bear or long_bear) and local_bear:
                    desired_signal = -BASE_SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals