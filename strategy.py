#!/usr/bin/env python3
"""
Experiment #004: 1d KAMA + RSI + Choppiness Regime

HYPOTHESIS: KAMA adapts to volatility automatically. In trending markets, KAMA 
stays above price for shorts/below for longs. RSI(14) confirms momentum. 
Choppiness Index filters out choppy periods. 1d timeframe naturally limits trades.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- KAMA adapts: fast in trending, slow in ranging (self-adjusting)
- Long: price > KAMA(10) + RSI > 55 + chop < 50
- Short: price < KAMA(10) + RSI < 45 + chop < 50
- ATR-based stops prevent large losses in crashes
- 1d = ~250 bars/year max = ~50-100 trades realistic

TARGET: 50-100 total trades over 4 years (proven pattern from DB).
DB reference: mtf_1d_kama_rsi_chop_regime_1w_v1 (Sharpe=1.310, 74tr)

KEY DESIGN:
1. KAMA(10) for trend detection
2. RSI(14) for momentum confirmation
3. Choppiness Index < 50 for trending regime
4. ATR(14) * 2.5 for stoploss
5. 1w KAMA aligned for higher timeframe confirmation
6. Signal: 0.25 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_rsi_chop_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=10, fast_ema=2, slow_ema=30):
    """
    Kaufman Adaptive Moving Average
    """
    n = len(close)
    if n < period + slow_ema:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                volatility += abs(close[j] - close[j - 1])
        
        if volatility > 1e-10:
            er[i] = price_change / volatility
    
    # Calculate smoothing constants
    fast_const = 2.0 / (fast_ema + 1)
    slow_const = 2.0 / (slow_ema + 1)
    scaling = (fast_const - slow_const) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan, dtype=np.float64)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if np.isnan(er[i]):
            continue
        sc = (er[i] * scaling + slow_const) ** 0.5 * (fast_const - slow_const) + slow_const
        if not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    deltas = np.diff(close, prepend=close[0])
    deltas[0] = 0
    
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.where(avg_loss > 1e-10, avg_gain / avg_loss, 0.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
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
    CHOP > 61.8 = ranging, CHOP < 50 = trending
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w data for HTF trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w KAMA for trend
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=10)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate 1d indicators
    kama_10 = calculate_kama(close, period=10)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    
    # Warmup (need 1w bars = ~4 1d bars)
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(kama_10[i]) or np.isnan(rsi_14[i]) or np.isnan(atr_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_1w_aligned[i]):
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
        is_trending = chop < 50.0  # Only trade in trending regime
        
        # === 1w TREND (HTF confirmation) ===
        price_above_1w_kama = close[i] > kama_1w_aligned[i]
        
        # === 1d KAMA TREND ===
        price_above_1d_kama = close[i] > kama_10[i]
        
        # === MOMENTUM ===
        rsi = rsi_14[i]
        rsi_bullish = rsi > 52.0
        rsi_bearish = rsi < 48.0
        
        # === VOLUME CONFIRMATION ===
        vol_ok = vol_ratio[i] > 0.9
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: Price above both KAMAs + RSI bullish + trending regime
        if is_trending:
            if price_above_1w_kama and price_above_1d_kama:
                if rsi_bullish and vol_ok:
                    desired_signal = SIZE
        
        # SHORT ENTRY: Price below both KAMAs + RSI bearish + trending regime
        if is_trending:
            if not price_above_1w_kama and not price_above_1d_kama:
                if rsi_bearish and vol_ok:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            stop_distance = entry_price - 2.5 * entry_atr
            if low[i] < stop_distance:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            stop_distance = entry_price + 2.5 * entry_atr
            if high[i] > stop_distance:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT (trailing stop, 2.5R) ===
        tp_triggered = False
        if in_position and position_side > 0:
            profit = (close[i] - entry_price) / entry_atr
            if profit >= 2.5:
                tp_triggered = True
        
        if in_position and position_side < 0:
            profit = (entry_price - close[i]) / entry_atr
            if profit >= 2.5:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
        
        signals[i] = desired_signal
    
    return signals