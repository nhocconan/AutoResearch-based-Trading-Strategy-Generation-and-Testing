#!/usr/bin/env python3
"""
Experiment #1636: 30m Primary + 4h HTF — HMA Trend + RSI Pullback + ROC Momentum

Hypothesis: 30m timeframe with 4h trend bias captures optimal entry timing while
maintaining HTF directionality. RSI pullback entries with ROC momentum confirmation
provide high-probability setups with manageable trade frequency.

LESSONS FROM FAILURES (#1625, #1629, #1630, #1633):
- Session filters (08-20 UTC) = ZERO trades (crypto trades 24/7!)
- RSI 40/60 too strict → use 35/65 for more entries
- Need momentum confirmation (ROC) that's easy to trigger
- Warmup too long (50 bars) → reduce to 30 bars
- Complex regime logic (Choppiness) reduces trade count

Key design choices:
1. NO session filter (crypto is 24/7 market)
2. LOOSE RSI thresholds: 35/65 (not 40/60) to guarantee ≥50 trades/year
3. 4h HMA(21) for trend bias (proven in best strategies)
4. ROC(10) momentum confirmation (easy to trigger, adds confluence)
5. Simple ATR(14) stoploss at 2.5x
6. Discrete signal sizes: 0.20 base, 0.30 strong
7. Short warmup (30 bars) to start trading earlier

Entry logic (LOOSE to guarantee trades):
- LONG: 4h HMA bullish + RSI(14) < 45 + ROC(10) > 0
- SHORT: 4h HMA bearish + RSI(14) > 55 + ROC(10) < 0
- Exit: RSI mean reversion (60/40) or stoploss hit

Why this beats failed 30m strategies:
- No session filter = trades throughout 24h crypto market
- Loose RSI + ROC = more entry opportunities
- 4h HMA = responsive trend filter without being too slow
- 3+ confluence (HMA + RSI + ROC) but each is loose

Target: Sharpe>0.6, trades≥50/year train, trades≥5 test, DD>-35%
Timeframe: 30m
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_roc_4h_loose_v1"
timeframe = "30m"
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

def calculate_roc(close, period=10):
    """Rate of Change - momentum indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = 100.0 * (close[i] - close[i - period]) / close[i - period]
    
    return roc

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    roc_10 = calculate_roc(close, period=10)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period (reduced from 50 to 30 for earlier trading)
    min_bars = 30
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(roc_10[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === RSI PULLBACK SIGNALS (LOOSE thresholds) ===
        rsi_val = rsi_14[i]
        
        # LONG: RSI pullback below 45 (loose)
        rsi_pullback_long = rsi_val < 45
        
        # SHORT: RSI pullback above 55 (loose)
        rsi_pullback_short = rsi_val > 55
        
        # === ROC MOMENTUM CONFIRMATION (easy to trigger) ===
        roc_val = roc_10[i]
        roc_bullish = roc_val > 0.0
        roc_bearish = roc_val < 0.0
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 4h HMA bullish + RSI pullback + ROC positive
        if price_above_4h and rsi_pullback_long and roc_bullish:
            desired_signal = SIZE_BASE
        
        # SHORT: 4h HMA bearish + RSI pullback + ROC negative
        elif price_below_4h and rsi_pullback_short and roc_bearish:
            desired_signal = -SIZE_BASE
        
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
        
        # === EXIT CONDITIONS (RSI mean reversion) ===
        # Exit long when RSI > 60 (overbought)
        if in_position and position_side > 0 and rsi_val > 60:
            desired_signal = 0.0
        
        # Exit short when RSI < 40 (oversold)
        if in_position and position_side < 0 and rsi_val < 40:
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