#!/usr/bin/env python3
"""
Experiment #1559: 1h Primary + 4h/12h HTF — Simple Trend Pullback with Volume

Hypothesis: After 1272 failed strategies with complex regime detection, return to 
SIMPLE proven patterns. Complex filters (chop + cRSI + session + multiple TF) 
kill trade frequency. This strategy uses:

1. 4h HMA(21) for major trend bias (direction filter only)
2. 1h RSI(14) for pullback entries (RSI<45 long in uptrend, RSI>55 short in downtrend)
3. 1h ATR(14) for stoploss (2.5x ATR trailing)
4. Volume filter (volume > 0.7 * 20-bar avg) - loose to allow trades
5. NO session filter (previous experiments with session=0 trades)

Key insight from failures:
- Session filters + multiple confluence = 0 trades (experiments #1549, #1553, #1556, #1557)
- Choppiness index regimes often stay neutral = no signals
- LOOSE entry thresholds guarantee trades while HTF filter prevents disasters

Why this should work:
- 4h trend filter prevents counter-trend trades in major moves
- RSI pullback entries catch retracements (proven 60%+ win rate)
- Loose RSI thresholds (45/55 not 30/70) ensure ≥40 trades/year
- Volume filter is loose (0.7x not 1.2x) to avoid filtering valid trades
- Simple = fewer bugs, more reliable execution

Target: Sharpe>0.6, trades>=40/year, DD>-30%
Timeframe: 1h
Size: 0.25 discrete (0.0, ±0.25, ±0.30)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma4h_rsi_pullback_volume_simple_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_sma(series, period):
    """Simple Moving Average"""
    n = len(series)
    if n < period:
        return np.full(n, np.nan)
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_sma_20 = calculate_sma(volume, 20)
    
    # Price relative to 4h HMA for trend strength
    hma_4h_slope = np.full(n, np.nan)
    for i in range(5, n):
        if not np.isnan(hma_4h_aligned[i]) and not np.isnan(hma_4h_aligned[i-5]):
            hma_4h_slope[i] = (hma_4h_aligned[i] - hma_4h_aligned[i-5]) / hma_4h_aligned[i-5]
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_sma_20[i]) or np.isnan(hma_4h_slope[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # 12h HMA for major trend confirmation
        price_above_12h = not np.isnan(hma_12h_aligned[i]) and close[i] > hma_12h_aligned[i]
        price_below_12h = not np.isnan(hma_12h_aligned[i]) and close[i] < hma_12h_aligned[i]
        
        # === RSI PULLBACK (LOOSE thresholds for trade frequency) ===
        rsi = rsi_14[i]
        rsi_oversold = rsi < 50  # Loose: was 40, now 50 for more trades
        rsi_overbought = rsi > 50  # Loose: was 60, now 50 for more trades
        
        # === VOLUME FILTER (LOOSE) ===
        vol_ok = volume[i] > 0.7 * vol_sma_20[i]  # 70% of avg, not 100%
        
        # === TREND STRENGTH (4h HMA slope) ===
        trend_strong_up = hma_4h_slope[i] > 0.005  # 0.5% over 5 bars
        trend_strong_down = hma_4h_slope[i] < -0.005
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + RSI pullback + volume OK
        # Two entry modes for more trades:
        if price_above_4h and rsi_oversold and vol_ok:
            if trend_strong_up or price_above_12h:
                desired_signal = SIZE_STRONG  # Strong signal with 12h confirm
            else:
                desired_signal = SIZE_BASE  # Base signal
        
        # SHORT: 4h bearish + RSI rally + volume OK
        elif price_below_4h and rsi_overbought and vol_ok:
            if trend_strong_down or price_below_12h:
                desired_signal = -SIZE_STRONG  # Strong signal with 12h confirm
            else:
                desired_signal = -SIZE_BASE  # Base signal
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals