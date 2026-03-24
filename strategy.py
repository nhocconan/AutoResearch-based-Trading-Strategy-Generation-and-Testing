#!/usr/bin/env python3
"""
Experiment #141: 4h Primary + 1d HTF — Supertrend + RSI Pullback + Volume Filter

Hypothesis: After 140+ experiments, the clearest pattern is:
- Supertrend provides cleaner trend signals than HMA/KAMA crossovers (less whipsaw)
- RSI pullback TO trend (not extremes) generates more trades than RSI extremes
- Volume confirmation filters out fake breakouts
- 1d HTF bias prevents counter-trend trades in strong trends
- Simple logic = more trades = better Sharpe (complex filters = 0 trades)

This strategy uses PROVEN components from literature:
1. Supertrend (ATR=10, mult=3) = primary trend direction
2. RSI(14) pullback = entry timing (RSI 35-45 for long, 55-65 for short)
3. Volume > SMA(20) = confirmation (avoids low-liquidity traps)
4. 1d Supertrend = major trend bias (only trade with HTF trend)
5. ATR(14) trailing stop 2.5x = risk management

Key design choices:
- Timeframe: 4h (proven 20-50 trades/year sweet spot)
- HTF: 1d Supertrend for bias (more responsive than 1w)
- RSI: 35-45/55-65 range (pullback zone, not extremes - ensures trades)
- Volume filter: > 20-bar SMA (confirms genuine moves)
- Position size: 0.28 (28% of capital, conservative)
- Stoploss: 2.5x ATR trailing (tighter for 4h timeframe)

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_supertrend_rsi_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Supertrend indicator - trend following with ATR bands
    Returns: supertrend values, direction (1=bull, -1=bear)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate basic bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Calculate Supertrend
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    direction = np.zeros(n)
    direction[:] = np.nan
    
    supertrend[period] = upper_band[period]
    direction[period] = -1  # Start bearish
    
    for i in range(period + 1, n):
        if direction[i-1] == 1:
            # Previous was bullish
            if lower_band[i] < supertrend[i-1]:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = supertrend[i-1]
            
            if close[i] < supertrend[i]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
        else:
            # Previous was bearish
            if upper_band[i] > supertrend[i-1]:
                supertrend[i] = upper_band[i]
            else:
                supertrend[i] = supertrend[i-1]
            
            if close[i] > supertrend[i]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
    
    return supertrend, direction

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
    """SMA of volume for volume filter"""
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d Supertrend for major trend bias
    _, st_dir_1d_raw = calculate_supertrend(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=10,
        multiplier=3.0
    )
    st_dir_1d_aligned = align_htf_to_ltf(prices, df_1d, st_dir_1d_raw)
    
    # Calculate primary (4h) indicators
    _, st_dir_4h = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (conservative for 4h)
    
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
        if np.isnan(st_dir_4h[i]) or np.isnan(st_dir_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d Supertrend direction) ===
        htf_bull = st_dir_1d_aligned[i] == 1
        htf_bear = st_dir_1d_aligned[i] == -1
        
        # === 4h TREND (Supertrend direction) ===
        st_bull = st_dir_4h[i] == 1
        st_bear = st_dir_4h[i] == -1
        
        # === RSI PULLBACK (loose thresholds for trade generation) ===
        # Long: RSI pulled back to 35-50 zone (not oversold, just cooling off)
        # Short: RSI pulled back to 50-65 zone (not overbought, just cooling off)
        rsi_pullback_long = 35.0 <= rsi[i] <= 50.0
        rsi_pullback_short = 50.0 <= rsi[i] <= 65.0
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > vol_sma[i]
        
        # === DESIRED SIGNAL ===
        # LONG: 1d bull + 4h Supertrend bull + RSI pullback + volume confirmed
        # SHORT: 1d bear + 4h Supertrend bear + RSI pullback + volume confirmed
        desired_signal = 0.0
        
        if htf_bull and st_bull and rsi_pullback_long and vol_confirmed:
            desired_signal = SIZE
        elif htf_bear and st_bear and rsi_pullback_short and vol_confirmed:
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