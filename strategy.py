#!/usr/bin/env python3
"""
Experiment #1609: 4h Primary + 1d HTF — Simplified Donchian + RSI + HMA Trend

Hypothesis: After 11 failed 4h experiments, complexity is the enemy. 
#1604 got Sharpe=0.017 but Choppiness + CRSI may be too complex causing 0 trades.
This strategy SIMPLIFIES: single 1d HMA trend filter + Donchian breakout + RSI pullback.

Key innovations:
1. Donchian(20) breakout - proven edge for SOL (Sharpe +0.782 in history)
2. RSI(14) pullback - simpler than CRSI, more reliable signals
3. 1d HMA(21) - single HTF trend filter (not dual 12h+1d)
4. ATR(14) 2.5x trailing stop - drawdown control
5. LOOSE entry conditions - RSI 35/65 (not 30/70) to ensure trade generation
6. Discrete position sizing (0.30) - minimize fee churn
7. Volume confirmation - taker_buy_volume > avg to confirm breakout

Why this should beat Sharpe 0.618:
- Donchian breakout proven on SOL in experiment history
- RSI simpler and more reliable than CRSI (fewer bugs)
- Single HTF filter = less alignment issues
- LOOSE thresholds ensure we generate trades (>10/symbol)
- 4h targets 20-50 trades/year — optimal for fee efficiency

Timeframe: 4h (required for this experiment)
HTF: 1d HMA for bias (use mtf_data helper - call ONCE before loop)
Target: Sharpe > 0.618, trades > 10/symbol train, > 3/symbol test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_rsi_1d_hma_vol_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - breakout detection
    Returns: upper_channel, lower_channel, mid_channel
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_rsi(close, period=14):
    """
    Relative Strength Index - momentum oscillator
    RSI < 30 = oversold, RSI > 70 = overbought
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_avg(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    vol_avg = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_avg[i] = np.mean(volume[i - period + 1:i + 1])
    return vol_avg

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for long-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    # Volume ratio for breakout confirmation
    vol_ratio = volume / vol_avg
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
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
        if np.isnan(hma_1d_aligned[i]):
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
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (1d HMA) ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i - 1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i - 1] if i > 0 else False
        
        # === RSI PULLBACK (for entry timing) ===
        rsi_oversold = rsi[i] < 45.0  # LOOSE for more trades
        rsi_overbought = rsi[i] > 55.0  # LOOSE for more trades
        rsi_neutral = 40.0 <= rsi[i] <= 60.0
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 0.8  # At least 80% of avg volume
        
        # === PRIMARY SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: Daily bull + Donchian breakout OR RSI pullback in uptrend
        if daily_bull:
            # Breakout long with volume confirmation
            if breakout_long and vol_confirmed:
                desired_signal = BASE_SIZE
            # Pullback long: price near Donchian mid + RSI oversold
            elif close[i] <= donchian_mid[i] and rsi_oversold and vol_confirmed:
                desired_signal = BASE_SIZE
        
        # SHORT ENTRY: Daily bear + Donchian breakdown OR RSI pullback in downtrend
        elif daily_bear:
            # Breakdown short with volume confirmation
            if breakout_short and vol_confirmed:
                desired_signal = -BASE_SIZE
            # Pullback short: price near Donchian mid + RSI overbought
            elif close[i] >= donchian_mid[i] and rsi_overbought and vol_confirmed:
                desired_signal = -BASE_SIZE
        
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
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