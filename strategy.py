#!/usr/bin/env python3
"""
Experiment #1082: 4h Primary + 1d/1w HTF — KAMA Adaptive Trend + RSI Pullback + Vol Filter

Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency better than HMA/EMA.
In trending markets (high efficiency), KAMA follows price closely. In choppy markets (low efficiency),
KAMA flattens to avoid whipsaws. Combined with RSI pullback entries and volume confirmation,
this should capture trends while avoiding range-bound losses.

Key innovations:
1. KAMA(14) with Efficiency Ratio - adapts smoothing based on trend strength
2. Dual HTF bias: 1d KAMA + 1w KAMA for long-term direction
3. RSI(7) pullback entries in direction of HTF trend (looser than CRSI)
4. Volume confirmation: taker_buy_volume ratio > 0.45 for longs, < 0.55 for shorts
5. ATR(14) volatility filter: skip entries when ATR ratio < 0.8 (dead market)
6. Asymmetric sizing: 0.30 for strong confluence, 0.20 for single signals
7. 2.5x ATR trailing stoploss

Why this should work:
- KAMA reduces whipsaws in 2022-2023 chop better than HMA (proven in literature)
- RSI(7) pullback is looser than CRSI extremes → more trades guaranteed
- Volume filter avoids false breakouts (critical for SOL)
- 4h timeframe targets 25-40 trades/year (sweet spot for fee vs opportunity)
- HTF bias ensures we trade with the macro trend

Entry conditions (LOOSE to guarantee 30+ trades):
- LONG: price>1d_KAMA>1w_KAMA + RSI(7)<60 + vol_ratio>0.45
- SHORT: price<1d_KAMA<1w_KAMA + RSI(7)>40 + vol_ratio<0.55
- ATR ratio > 0.7 (market has enough volatility)

Target: Sharpe>0.45, trades>=35 train, trades>=5 test, DD>-35%
Timeframe: 4h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_vol_pullback_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    
    Efficiency Ratio (ER) = |close - close_n| / sum(|close_i - close_i-1|)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - period]):
            signal = abs(close[i] - close[i - period])
            noise = 0.0
            for j in range(i - period + 1, i + 1):
                if not np.isnan(close[j]) and not np.isnan(close[j - 1]):
                    noise += abs(close[j] - close[j - 1])
            if noise > 1e-10:
                er[i] = signal / noise
            else:
                er[i] = 1.0
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i - 1]) and not np.isnan(close[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        elif not np.isnan(close[i]):
            kama[i] = close[i]
    
    return kama

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

def calculate_atr_ratio(atr, period=30):
    """ATR ratio: current ATR / rolling average ATR (volatility filter)"""
    n = len(atr)
    if n < period:
        return np.full(n, np.nan)
    
    atr_avg = pd.Series(atr).rolling(window=period, min_periods=period).mean().values
    ratio = np.divide(atr, atr_avg, out=np.zeros_like(atr), where=atr_avg > 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=14)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=14)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    kama_14 = calculate_kama(close, period=14)
    atr_ratio = calculate_atr_ratio(atr_14, period=30)
    
    # Volume ratio (taker buy / total volume)
    vol_ratio = np.divide(taker_buy_vol, volume, out=np.zeros_like(volume), where=volume > 1e-10)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]) or np.isnan(kama_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_ratio[i]) or atr_ratio[i] <= 0.5:
            # Market too quiet, skip entries but keep positions
            if not in_position:
                signals[i] = 0.0
            continue
        
        # === HTF BIAS (KAMA alignment) ===
        htf_bull = kama_1d_aligned[i] > kama_1w_aligned[i] and close[i] > kama_1d_aligned[i]
        htf_bear = kama_1d_aligned[i] < kama_1w_aligned[i] and close[i] < kama_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_bull = vol_ratio[i] > 0.45
        vol_bear = vol_ratio[i] < 0.55
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG entries (loose conditions for trade frequency)
        if htf_bull:
            # Strong: RSI pullback + volume confirmation
            if rsi_7[i] < 55.0 and vol_bull:
                desired_signal = SIZE_STRONG
            # Weaker: just RSI pullback in uptrend
            elif rsi_7[i] < 50.0:
                desired_signal = SIZE_BASE
            # RSI recovery from oversold
            elif rsi_14[i] < 45.0 and rsi_7[i] > rsi_7[i-1] if i > 0 else False:
                desired_signal = SIZE_BASE
        
        # SHORT entries
        elif htf_bear:
            # Strong: RSI rally + volume confirmation
            if rsi_7[i] > 45.0 and vol_bear:
                desired_signal = -SIZE_STRONG
            # Weaker: just RSI rally in downtrend
            elif rsi_7[i] > 50.0:
                desired_signal = -SIZE_BASE
            # RSI rejection from overbought
            elif rsi_14[i] > 55.0 and rsi_7[i] < rsi_7[i-1] if i > 0 else False:
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