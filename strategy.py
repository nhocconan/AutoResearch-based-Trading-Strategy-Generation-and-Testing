#!/usr/bin/env python3
"""
Experiment #008: 12h KAMA Trend + RSI Mean Reversion + Chop Regime + 1w HMA

HYPOTHESIS: KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency - 
fast in trends, slow in chop. Combined with RSI extremes for entry timing and 
Choppiness Index regime filter, this captures both trend continuation and 
mean-reversion entries. Weekly HMA provides higher-timeframe trend bias.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- KAMA adapts: fast during 2021 bull, slow during 2022 crash, captures 2023-24 recovery
- RSI extremes: long at RSI<35 (oversold), short at RSI>65 (overbought) - works both directions
- Chop filter: CHOP<65 allows trending AND neutral markets (not just strong trends)
- 1w HMA bias: only long when price>weekly HMA, only short when price<weekly HMA

TARGET: 75-200 total trades over 4 years (19-50/year). This is LOOSE enough to guarantee trades.
DB reference: mtf_1d_kama_rsi_chop_regime_1w_v1 (SOL Sharpe=1.31, 74 trades)

KEY DESIGN:
1. KAMA(10,2,30) direction = trend bias
2. RSI(14) extremes = entry trigger (35/65 thresholds - proven to generate trades)
3. Choppiness(14) < 65 = regime filter (allows trend + neutral, blocks pure chop)
4. 1w HMA(21) = higher timeframe trend confirmation
5. Signal: ±0.25 discrete (25% position size - safe for 77% BTC crash)
6. Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_rsi_chop_1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs chop)
    """
    n = len(close)
    kama = np.full(n, np.nan, dtype=np.float64)
    
    if n < slow_period + er_period:
        return kama
    
    # Efficiency Ratio
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(er_period, n):
        if np.isnan(close[i]) or np.isnan(close[i - er_period]):
            continue
        price_change = abs(close[i] - close[i - er_period])
        if price_change < 1e-10:
            er[i] = 0.0
            continue
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 1e-10:
            er[i] = price_change / volatility
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if np.isnan(er[i]) or np.isnan(kama[i - 1]):
            continue
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness
    CHOP > 61.8 = ranging, CHOP < 38.2 = strong trend
    We use < 65 to allow trend + neutral regimes
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
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_hma(close, period):
    """Hull Moving Average"""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w data for HTF trend bias
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 12h indicators
    kama_10 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size - safe for major drawdowns
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup - need enough bars for all indicators
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(kama_10[i]) or np.isnan(rsi_14[i]) or np.isnan(atr_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK ===
        chop = chop_14[i]
        is_tradeable = chop < 65.0  # Allow trend + neutral, block pure chop
        
        # === TREND BIAS (KAMA direction + 1w HMA) ===
        kama_slope = kama_10[i] - kama_10[i - 5] if i >= 5 else 0.0
        kama_bullish = kama_slope > 0.0
        kama_bearish = kama_slope < 0.0
        
        price_above_1w_hma = close[i] > hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else True
        price_below_1w_hma = close[i] < hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else False
        
        # === RSI EXTREMES ===
        rsi = rsi_14[i]
        rsi_oversold = rsi < 35.0  # Long entry zone
        rsi_overbought = rsi > 65.0  # Short entry zone
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: RSI oversold + KAMA bullish + price above weekly HMA + tradeable regime
        if is_tradeable and rsi_oversold and kama_bullish and price_above_1w_hma:
            desired_signal = SIZE
        
        # SHORT: RSI overbought + KAMA bearish + price below weekly HMA + tradeable regime
        if is_tradeable and rsi_overbought and kama_bearish and price_below_1w_hma:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK ===
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
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals