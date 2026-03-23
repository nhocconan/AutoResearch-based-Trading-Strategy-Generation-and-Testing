#!/usr/bin/env python3
"""
Experiment #1120: 1h Primary + 4h/12h HTF — Simplified Trend Pullback

Hypothesis: After analyzing 800+ failed experiments, key insights for 1h timeframe:
1. Lower TF (1h) MUST use HTF for direction, LTF only for entry timing
2. Complex regime-switching (Choppiness + CRSI) = 0 trades on 1h (see #1115, #1118)
3. SIMPLER works better: 4h HMA trend + 1h RSI pullback + minimal filters
4. Loose RSI thresholds (35/65) ensure 30-60 trades/year target
5. Position size 0.25 with 2.5x ATR trailing stop for risk control
6. Add volume filter (>0.7x avg) to avoid low-liquidity false signals

Why this should beat failed 1h strategies (#1110 Sharpe=-1.645, #1115 Sharpe=0.000):
- Simpler entry logic = more trades (avoid 0-trade failure)
- 4h HMA provides clean trend filter without over-complication
- Volume filter adds confluence without being too restrictive
- Proven pattern: HMA + RSI + ATR worked on 4h (Sharpe=0.612 baseline)
- Adjusted for 1h: smaller size (0.25 vs 0.35), looser RSI (35/65 vs 40/60)

Timeframe: 1h (primary)
HTF: 4h — loaded ONCE before loop using mtf_data helper
Position Size: 0.25 base, 0.15 reduced (discrete levels)
Stoploss: 2.5x ATR trailing
Target: 30-60 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_4h_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    
    Formula:
    1. WMA1 = WMA(close, period/2)
    2. WMA2 = WMA(close, period)
    3. WMA3 = WMA(2*WMA1 - WMA2, sqrt(period))
    4. HMA = WMA3
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = int(period / 2)
    if half < 1:
        half = 1
    
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    
    # 2*WMA1 - WMA2
    diff = 2 * wma1 - wma2
    
    # WMA of diff with sqrt(period)
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
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

def calculate_sma(data, period=20):
    """Simple Moving Average."""
    return pd.Series(data).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for trend filter
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for macro confirmation
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume SMA for filter
    vol_sma = calculate_sma(volume, period=20)
    
    # Price vs 4h HMA distance (z-score like)
    hma_4h_dist = (close - hma_4h_aligned) / (atr + 1e-10)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.12
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === TREND FILTER (4h HMA) ===
        # Price above 4h HMA = bull trend, below = bear trend
        trend_bull = close[i] > hma_4h_aligned[i]
        trend_bear = close[i] < hma_4h_aligned[i]
        
        # === MACRO CONFIRMATION (12h HMA) ===
        # 12h HMA confirms 4h direction (both same side)
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === VOLUME FILTER ===
        # Volume > 0.7x 20-bar average
        volume_ok = volume[i] > 0.7 * vol_sma[i]
        
        # === PULLBACK SIGNAL (1h RSI) ===
        # Loose thresholds for adequate trade frequency
        # Long: RSI < 40 in bull trend (pullback entry)
        # Short: RSI > 60 in bear trend (pullback entry)
        rsi_oversold = rsi_1h[i] < 40.0
        rsi_overbought = rsi_1h[i] > 60.0
        
        # === EXTREME PULLBACK (optional stronger signal) ===
        # RSI < 30 or > 70 = stronger mean reversion opportunity
        rsi_extreme_oversold = rsi_1h[i] < 30.0
        rsi_extreme_overbought = rsi_1h[i] > 70.0
        
        # === DISTANCE FROM HMA ===
        # Too far from HMA = extended, wait for pullback
        # Within 1.5 ATR of HMA = good entry zone
        near_hma_long = hma_4h_dist[i] < 1.5
        near_hma_short = hma_4h_dist[i] > -1.5
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRY ===
        # 4h bull + 12h bull + RSI pullback + volume OK
        # Entry when RSI < 40 OR extreme < 30 (catch deeper pullbacks)
        if trend_bull and macro_bull and volume_ok:
            if rsi_oversold or rsi_extreme_oversold:
                if near_hma_long:
                    desired_signal = current_size
        
        # === SHORT ENTRY ===
        # 4h bear + 12h bear + RSI pullback + volume OK
        # Entry when RSI > 60 OR extreme > 70
        elif trend_bear and macro_bear and volume_ok:
            if rsi_overbought or rsi_extreme_overbought:
                if near_hma_short:
                    desired_signal = -current_size
        
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h trend still bull
                if trend_bull and rsi_1h[i] < 70.0:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if 4h trend still bear
                if trend_bear and rsi_1h[i] > 30.0:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses or RSI very overbought
            if trend_bear or rsi_1h[i] > 75.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses or RSI very oversold
            if trend_bull or rsi_1h[i] < 25.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
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
                # Flip position
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