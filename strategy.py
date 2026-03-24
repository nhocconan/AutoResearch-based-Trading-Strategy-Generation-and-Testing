#!/usr/bin/env python3
"""
Experiment #1048: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + RSI Pullback + Volume Filter

Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to market volatility better than HMA/EMA,
reducing whipsaws in choppy markets while capturing trends efficiently. Combined with 12h trend bias
and RSI pullback entries on 4h, this should generate consistent signals with fewer false entries.

Key innovations:
1. KAMA (Efficiency Ratio = 10): Adapts smoothing based on market noise
   - Fast SC = 2/(2+1) = 0.667, Slow SC = 2/(30+1) = 0.065
   - ER = |close - close_n| / sum(|close_i - close_i-1|)
2. 12h HMA(21) for intermediate trend bias (faster than 1d, slower than 4h)
3. 1d HMA(21) for long-term bias filter
4. RSI(14) pullback entries: Long when RSI 40-55 in uptrend, Short when RSI 45-60 in downtrend
5. Volume filter: Only enter when volume > 0.8 * volume_SMA(20)
6. ATR(14) 2.0x trailing stop (tighter than 2.5x to reduce drawdown)
7. Discrete sizing: 0.0, ±0.25, ±0.30

Why this should work:
- KAMA reduces lag in trends while smoothing in ranges (adaptive)
- 12h HTF is sweet spot between 4h noise and 1d slowness
- RSI pullback (not extreme) entries catch continuation moves
- Volume filter avoids low-liquidity false breakouts
- 4h timeframe targets 25-40 trades/year (fee-efficient)

Entry conditions (LOOSE to guarantee trades):
- LONG: price > 12h_HMA > 1d_HMA + RSI(14) 40-55 + volume > 0.8*vol_SMA
- SHORT: price < 12h_HMA < 1d_HMA + RSI(14) 45-60 + volume > 0.8*vol_SMA
- Exit: RSI crosses opposite threshold OR stoploss hit

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_pullback_vol_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency/volatility
    
    Efficiency Ratio (ER) = |Close - Close_n| / Sum(|Close_i - Close_i-1|)
    Smoothing Constant (SC) = ER * (fast_SC - slow_SC) + slow_SC
    fast_SC = 2/(fast_period+1), slow_SC = 2/(slow_period+1)
    
    KAMA[i] = KAMA[i-1] + SC * (Close[i] - KAMA[i-1])
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA with SMA of first period bars
    kama[period - 1] = np.nanmean(close[:period])
    
    for i in range(period, n):
        # Efficiency Ratio: signal / noise
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        
        if noise > 1e-10:
            er = signal / noise
        else:
            er = 0.0
        
        # Smoothing constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    kama_10 = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume SMA for filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_10[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS ===
        # 12h HMA vs 1d HMA alignment
        hma_12h_above_1d = hma_12h_aligned[i] > hma_1d_aligned[i]
        hma_12h_below_1d = hma_12h_aligned[i] < hma_1d_aligned[i]
        
        # Price vs 12h HMA
        price_above_12h = close[i] > hma_12h_aligned[i]
        price_below_12h = close[i] < hma_12h_aligned[i]
        
        # Price vs KAMA
        price_above_kama = close[i] > kama_10[i]
        price_below_kama = close[i] < kama_10[i]
        
        # KAMA slope (simplified: current vs 5 bars ago)
        kama_slope_up = not np.isnan(kama_10[i-5]) and kama_10[i] > kama_10[i-5]
        kama_slope_down = not np.isnan(kama_10[i-5]) and kama_10[i] < kama_10[i-5]
        
        # === VOLUME FILTER ===
        vol_confirmed = volume[i] > 0.8 * vol_sma[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Uptrend + RSI pullback + volume
        if price_above_12h and hma_12h_above_1d and price_above_kama and kama_slope_up:
            if 40.0 <= rsi_14[i] <= 55.0 and vol_confirmed:
                desired_signal = SIZE_BASE
            elif 45.0 <= rsi_14[i] <= 52.0 and vol_confirmed:
                desired_signal = SIZE_STRONG
        
        # SHORT: Downtrend + RSI pullback + volume
        elif price_below_12h and hma_12h_below_1d and price_below_kama and kama_slope_down:
            if 45.0 <= rsi_14[i] <= 60.0 and vol_confirmed:
                desired_signal = -SIZE_BASE
            elif 48.0 <= rsi_14[i] <= 55.0 and vol_confirmed:
                desired_signal = -SIZE_STRONG
        
        # === STOPLOSS CHECK (2.0x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT ON RSI REVERSAL ===
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            desired_signal = 0.0  # Overbought exit
        elif in_position and position_side < 0 and rsi_14[i] < 30.0:
            desired_signal = 0.0  # Oversold exit
        
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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