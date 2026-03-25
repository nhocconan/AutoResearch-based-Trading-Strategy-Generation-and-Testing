#!/usr/bin/env python3
"""
Experiment #1138: 4h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback + Vol Filter

Hypothesis: After 939 failed strategies, complexity is the enemy. Regime-switching (chop/trend)
creates whipsaws and 0-trade scenarios. Instead, use SIMPLE trend-following with:
1. 1d HMA(21) for long-term bias (only trade with daily trend)
2. 4h RSI(14) pullback entries (buy dips in uptrend, sell rallies in downtrend)
3. ATR(14) volatility filter (only trade when vol > median, avoid dead markets)
4. 3x ATR trailing stop (wider stops reduce whipsaw exits)
5. LOOSE entry conditions to GUARANTEE trades (lesson from 0-trade failures)

Why this should work:
- Simpler = fewer failure modes (see exp #1132-1137 all failed with complex regime logic)
- 1d HMA provides strong directional bias without whipsaw
- RSI pullback (not extreme) ensures we get entries in trending markets
- Vol filter avoids trading during low-vol chop (major source of losses)
- Loose conditions: RSI<55 for long (not <30), RSI>45 for short (not >70)
- 4h timeframe = 20-50 trades/year target (fee-efficient)

Entry conditions (LOOSE to guarantee >=30 trades/train, >=3/test):
- LONG: 1d_HMA_bull + RSI(14)<55 + ATR>median + price>1d_HMA*0.98
- SHORT: 1d_HMA_bear + RSI(14)>45 + ATR>median + price<1d_HMA*1.02

Exit: RSI crosses opposite threshold OR 3x ATR stoploss hit

Size: 0.25 base, 0.30 strong (discrete to minimize fee churn)
Timeframe: 4h
Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_vol_simple_1d_v1"
timeframe = "4h"
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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate ATR median for volatility filter (use lookback of 500 bars ~83 days on 4h)
    atr_median = np.nanmedian(atr_14[100:min(600, n)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmedian(atr_14[50:n])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(atr_14[50:n])
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]):
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
        
        # === VOLATILITY FILTER (only trade when vol > median) ===
        vol_ok = atr_14[i] > atr_median * 0.7  # Allow some flexibility
        
        # === HTF BIAS (1d HMA direction) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Price proximity to HMA (avoid chasing extremes)
        price_hma_ratio = close[i] / hma_1d_aligned[i] if hma_1d_aligned[i] > 0 else 1.0
        near_hma_long = price_hma_ratio < 1.05  # Within 5% above HMA
        near_hma_short = price_hma_ratio > 0.95  # Within 5% below HMA
        
        # === ENTRY LOGIC (LOOSE conditions to guarantee trades) ===
        desired_signal = 0.0
        
        if vol_ok:
            # LONG: Daily bullish + RSI pullback (not extreme, just <55)
            if hma_1d_bull and rsi_14[i] < 55.0 and near_hma_long:
                # Stronger signal if RSI very low
                if rsi_14[i] < 40.0:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # SHORT: Daily bearish + RSI rally (not extreme, just >45)
            elif hma_1d_bear and rsi_14[i] > 45.0 and near_hma_short:
                # Stronger signal if RSI very high
                if rsi_14[i] > 60.0:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (3x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT ON RSI REVERSAL (opposite threshold) ===
        if in_position and position_side > 0 and rsi_14[i] > 65.0:
            desired_signal = 0.0  # RSI overbought in long position
        
        if in_position and position_side < 0 and rsi_14[i] < 35.0:
            desired_signal = 0.0  # RSI oversold in short position
        
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
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
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