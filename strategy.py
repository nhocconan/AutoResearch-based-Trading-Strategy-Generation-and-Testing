#!/usr/bin/env python3
"""
Experiment #230: 1h Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Volume Confirm

Hypothesis: After repeated failures with complex regime-switching (#220, #225, #227),
simplify to proven multi-timeframe trend-following with mean-reversion entries.
Key insight from failures: too many filters = 0 trades (30m strategies returned 0%).

Strategy logic:
1. 12h HMA(21) = macro trend bias (long only above, short only below)
2. 4h HMA(16/48) = intermediate trend confirmation
3. 1h RSI(14) pullback = entry timing (buy dips in uptrend, sell rallies in downtrend)
4. Volume spike = confirmation (volume > 1.2x 20-bar avg)
5. ATR(14)*2.5 = trailing stoploss

CRITICAL: Relaxed RSI thresholds (30/70 instead of 35/65) to ensure trade frequency.
Target: 40-70 trades/year on 1h timeframe (within 30-60 target range).
Position sizing: 0.0, ±0.20, ±0.30 (discrete levels to minimize fee churn).

Why this might work:
- 12h trend filter prevents counter-trend trades (major source of losses in 2022)
- RSI pullback entries catch better prices than breakout entries
- Volume confirmation filters false signals
- ATR stoploss protects against black swan events
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_vol_4h12h_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) window
    Faster and smoother than EMA, less lag.
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, window):
        if window < 1:
            return series
        weights = np.arange(1, window + 1, dtype=float)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_volume_spike(volume, period=20, threshold=1.2):
    """Detect volume spikes above moving average."""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    spike = volume > (threshold * vol_ma)
    return spike

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1h indicators (primary timeframe)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_spike = calculate_volume_spike(volume, period=20, threshold=1.2)
    
    # Calculate 4h HMA trend (aligned properly)
    hma_4h_16_raw = calculate_hma(df_4h['close'].values, 16)
    hma_4h_48_raw = calculate_hma(df_4h['close'].values, 48)
    hma_4h_16 = align_htf_to_ltf(prices, df_4h, hma_4h_16_raw)
    hma_4h_48 = align_htf_to_ltf(prices, df_4h, hma_4h_48_raw)
    
    # Calculate 12h HMA for macro bias (aligned properly)
    hma_12h_21_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_21 = align_htf_to_ltf(prices, df_12h, hma_12h_21_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            continue
        if np.isnan(hma_12h_21[i]):
            continue
        
        # === MACRO TREND BIAS (12h HMA) ===
        price_above_hma_12h = close[i] > hma_12h_21[i]
        price_below_hma_12h = close[i] < hma_12h_21[i]
        
        # === INTERMEDIATE TREND (4h HMA crossover) ===
        hma_4h_bullish = hma_4h_16[i] > hma_4h_48[i]
        hma_4h_bearish = hma_4h_16[i] < hma_4h_48[i]
        
        # === RSI EXTREMES (relaxed thresholds for trade frequency) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_neutral_long = 35.0 <= rsi_14[i] <= 55.0
        rsi_neutral_short = 45.0 <= rsi_14[i] <= 65.0
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_spike[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: 12h bullish + 4h bullish + RSI pullback + volume confirm
        if price_above_hma_12h and hma_4h_bullish:
            if rsi_oversold:
                # Strong oversold entry
                new_signal = POSITION_SIZE_FULL
            elif rsi_neutral_long and vol_confirmed:
                # Pullback entry with volume
                new_signal = POSITION_SIZE_HALF
        
        # SHORT ENTRY: 12h bearish + 4h bearish + RSI rally + volume confirm
        elif price_below_hma_12h and hma_4h_bearish:
            if rsi_overbought:
                # Strong overbought entry
                new_signal = -POSITION_SIZE_FULL
            elif rsi_neutral_short and vol_confirmed:
                # Rally entry with volume
                new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if trends still valid and RSI not extreme
                if price_above_hma_12h and hma_4h_bullish and rsi_14[i] < 75.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if trends still valid and RSI not extreme
                if price_below_hma_12h and hma_4h_bearish and rsi_14[i] > 25.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side > 0:
            if price_below_hma_12h or hma_4h_bearish:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_12h or hma_4h_bullish:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals