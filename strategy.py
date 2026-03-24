#!/usr/bin/env python3
"""
Experiment #1067: 6h Primary + 1d HTF — KAMA Adaptive Trend + Volatility Regime + RSI Pullback

Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to market efficiency better than EMA/HMA,
reducing whipsaws in choppy markets while capturing trends efficiently. Combined with volatility
regime detection (ATR ratio) and RSI pullback entries, this should outperform static MA strategies.

Key innovations:
1. KAMA (Kaufman Adaptive MA): Efficiency Ratio adapts smoothing constant - fast in trends, slow in noise
2. Volatility Regime: ATR(7)/ATR(30) ratio detects vol spikes (>2.0) vs compression (<1.0)
3. Regime-adaptive entries:
   - High vol (ATR ratio > 1.8): Mean reversion (fade extremes with RSI)
   - Low vol (ATR ratio < 1.2): Trend following (KAMA slope + HTF bias)
   - Normal vol: Mixed strategy with tighter filters
4. 1d HMA(21) for higher timeframe directional bias
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work:
- KAMA reduces whipsaws vs EMA/HMA in range markets (2022-2023)
- Volatility regime detection avoids trend-following during vol spikes (panic bottoms)
- 6h captures multi-day swings without 4h noise or 12h slowness
- 1d HTF bias prevents counter-trend trades in strong trends
- Loose entry conditions guarantee 30-60 trades/year target

Entry conditions (LOOSE to guarantee trades):
- LONG high-vol: ATR_ratio>1.8 + RSI<35 + price>1d_HMA*0.92
- LONG low-vol: ATR_ratio<1.2 + KAMA_slope>0 + price>1d_HMA + RSI>45
- LONG normal: price>KAMA + RSI>50 + 1d_HMA_bull + CHOP<50
- SHORT high-vol: ATR_ratio>1.8 + RSI>65 + price<1d_HMA*1.08
- SHORT low-vol: ATR_ratio<1.2 + KAMA_slope<0 + price<1d_HMA + RSI<55
- SHORT normal: price<KAMA + RSI<50 + 1d_HMA_bear + CHOP<50

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_volregime_rsi_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs noise)
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

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
    
    # Calculate 6h indicators
    kama_10 = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volatility ratio (ATR 7 / ATR 30)
    atr_ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(30, n):
        if atr_30[i] > 1e-10 and not np.isnan(atr_7[i]):
            atr_ratio[i] = atr_7[i] / atr_30[i]
    
    # KAMA slope (rate of change)
    kama_slope = np.full(n, np.nan, dtype=np.float64)
    for i in range(11, n):
        if not np.isnan(kama_10[i]) and not np.isnan(kama_10[i-1]) and kama_10[i-1] > 1e-10:
            kama_slope[i] = (kama_10[i] - kama_10[i-1]) / kama_10[i-1] * 1000
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(kama_10[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_ratio[i]) or np.isnan(kama_slope[i]):
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
        
        # === VOLATILITY REGIME DETECTION ===
        high_vol = atr_ratio[i] > 1.8  # Vol spike - mean reversion mode
        low_vol = atr_ratio[i] < 1.2   # Vol compression - trend mode
        # normal_vol = not high_vol and not low_vol
        
        # === HTF BIAS (1d HMA) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i] * 0.98  # Slight buffer
        hma_1d_bear = close[i] < hma_1d_aligned[i] * 1.02
        
        # === KAMA TREND ===
        kama_bull = close[i] > kama_10[i] and kama_slope[i] > 0
        kama_bear = close[i] < kama_10[i] and kama_slope[i] < 0
        
        # === ENTRY LOGIC (VOLATILITY-ADAPTIVE) ===
        desired_signal = 0.0
        
        if high_vol:
            # HIGH VOLATILITY - Mean Reversion (fade extremes)
            # Long when RSI oversold + price near 1d HMA support
            if rsi_14[i] < 38.0 and close[i] > hma_1d_aligned[i] * 0.90:
                desired_signal = SIZE_BASE
            elif rsi_14[i] < 30.0 and close[i] > hma_1d_aligned[i] * 0.88:
                desired_signal = SIZE_STRONG
            # Short when RSI overbought + price near 1d HMA resistance
            elif rsi_14[i] > 62.0 and close[i] < hma_1d_aligned[i] * 1.10:
                desired_signal = -SIZE_BASE
            elif rsi_14[i] > 70.0 and close[i] < hma_1d_aligned[i] * 1.12:
                desired_signal = -SIZE_STRONG
        
        elif low_vol:
            # LOW VOLATILITY - Trend Following (KAMA + HTF bias)
            # Long in uptrend with KAMA confirmation
            if kama_bull and hma_1d_bull and rsi_14[i] > 45.0 and rsi_14[i] < 75.0:
                desired_signal = SIZE_STRONG
            elif kama_bull and hma_1d_bull and rsi_14[i] > 50.0:
                desired_signal = SIZE_BASE
            # Short in downtrend with KAMA confirmation
            elif kama_bear and hma_1d_bear and rsi_14[i] < 55.0 and rsi_14[i] > 25.0:
                desired_signal = -SIZE_STRONG
            elif kama_bear and hma_1d_bear and rsi_14[i] < 50.0:
                desired_signal = -SIZE_BASE
        
        else:
            # NORMAL VOLATILITY - Mixed strategy with CHOP filter
            # Long when trend + not too choppy
            if kama_bull and hma_1d_bull and chop_14[i] < 55.0 and rsi_14[i] > 48.0:
                desired_signal = SIZE_BASE
            # Short when trend + not too choppy
            elif kama_bear and hma_1d_bear and chop_14[i] < 55.0 and rsi_14[i] < 52.0:
                desired_signal = -SIZE_BASE
            # Mean reversion in choppy market
            elif chop_14[i] > 55.0 and rsi_14[i] < 35.0:
                desired_signal = SIZE_BASE
            elif chop_14[i] > 55.0 and rsi_14[i] > 65.0:
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