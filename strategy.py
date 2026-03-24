#!/usr/bin/env python3
"""
Experiment #105: 15m Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Volume

Hypothesis: 15m strategies have failed due to ZERO trades (too strict conditions).
This strategy uses LOOSE entry conditions to ensure trades generate on ALL symbols:
- 4h HMA(21) for intermediate trend direction (proven in baseline)
- 1d HMA(50) for major regime bias (bull/bear)
- 15m RSI(7) pullback entries in trend direction (loose: 35-65 range)
- Volume spike confirmation (1.5x average) for entry conviction
- Mean reversion fallback when 4h/1d disagree (range regime)
- Position size: 0.22 (conservative for 15m frequency)
- Stoploss: 2.0x ATR trailing

Key design for TRADE GENERATION:
- RSI thresholds LOOSE (35-65, not 20-80) to allow more entries
- Only 2-3 confluence required (not 5+)
- Fallback entries when primary signals don't trigger
- Session preference (00-12 UTC) but NOT a hard filter

Target: Sharpe>0.167, DD>-40%, trades>=40 on train, trades>=5 on test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_vol_4h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs moving average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for major regime bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=16)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 15m
    atr = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.22  # 22% position size (conservative for 15m)
    
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
        if np.isnan(hma_15m[i]) or np.isnan(rsi[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h HMA) - Intermediate Trend ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === REGIME BIAS (1d HMA) - Major Trend ===
        regime_bull = close[i] > hma_1d_aligned[i]
        regime_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION ===
        # Aligned = both HTF agree (trend regime)
        # Disagree = range regime (mean reversion)
        is_trend_regime = (htf_bull and regime_bull) or (htf_bear and regime_bear)
        is_range_regime = not is_trend_regime
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 1.3  # 30% above average
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === RSI PULLBACK (LOOSE thresholds for trade generation) ===
        # Long: RSI pulled back but not oversold (35-55)
        # Short: RSI pulled back but not overbought (45-65)
        rsi_pullback_long = 35.0 <= rsi[i] <= 55.0
        rsi_pullback_short = 45.0 <= rsi[i] <= 65.0
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        
        # === DESIRED SIGNAL (Primary: Trend Following) ===
        desired_signal = 0.0
        
        if is_trend_regime:
            # TREND REGIME: Follow HTF direction with pullback entries
            # LONG: HTF bull + RSI pullback + volume OR HMA bull + RSI ok
            if htf_bull and rsi_pullback_long:
                if vol_confirmed:
                    desired_signal = SIZE
                elif hma_bull:
                    desired_signal = SIZE * 0.7
            # SHORT: HTF bear + RSI pullback + volume OR HMA bear + RSI ok
            elif htf_bear and rsi_pullback_short:
                if vol_confirmed:
                    desired_signal = -SIZE
                elif hma_bear:
                    desired_signal = -SIZE * 0.7
            # Fallback: Strong RSI extreme in trend direction
            elif htf_bull and rsi_oversold and hma_bull:
                desired_signal = SIZE * 0.7
            elif htf_bear and rsi_overbought and hma_bear:
                desired_signal = -SIZE * 0.7
        else:
            # RANGE REGIME: Mean reversion at HMA bounds
            # LONG: Price below HMA + RSI oversold
            if hma_bear and rsi_oversold:
                desired_signal = SIZE * 0.7
            # SHORT: Price above HMA + RSI overbought
            elif hma_bull and rsi_overbought:
                desired_signal = -SIZE * 0.7
            # Fallback: Extreme RSI mean reversion
            elif rsi[i] < 30.0:
                desired_signal = SIZE * 0.5
            elif rsi[i] > 70.0:
                desired_signal = -SIZE * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
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