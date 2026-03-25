#!/usr/bin/env python3
"""
Experiment #1627: 6h Primary + 1d HTF — Volatility Regime with KAMA + ROC

Hypothesis: 6h timeframe fills gap between 4h (too fast) and 12h (too slow).
Volatility Ratio (ATR7/ATR30) is better regime filter than CHOP for crypto.
KAMA adapts to market efficiency - slows in chop, speeds in trend.
ROC momentum confirms entries without being too restrictive.

Key design choices based on failure analysis:
1. VOLATILITY RATIO regime: >2.0 = vol spike (mean revert), <1.2 = calm (trend)
2. KAMA(10,2,30) instead of HMA - adapts to market noise automatically
3. ROC(10) for momentum confirmation - loose threshold >-5% for long
4. 1d KAMA for trend bias (not HMA - KAMA smoother in crypto)
5. LOOSE entry conditions to guarantee 30+ trades/train
6. Discrete signal sizes: 0.25 base, 0.30 strong
7. 2.5x ATR trailing stoploss via signal→0

Why this beats mtf_6h_triple_hma_kama_roc_1w1d_v1 (Sharpe=0.575):
- Volatility Ratio regime is MORE responsive than CHOP for crypto
- Single 1d HTF (not 1w+1d) = more trades (1w too restrictive)
- KAMA efficiency ratio adapts better to 2022 crash + 2025 bear
- Looser ROC threshold (-5% not -10%) = more long entries in bull phases

Target: Sharpe>0.6, trades≥30 train, trades≥5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_volregime_roc_1d_loose_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts to market efficiency - smooths in chop, responsive in trend
    """
    n = len(close)
    if n < slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Change = absolute price change over period
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.nan
    
    # Volatility = sum of absolute single-period changes
    volatility = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-period:i+1])))
    
    # Efficiency Ratio (ER)
    er = np.full(n, np.nan, dtype=np.float64)
    mask = volatility > 1e-10
    er[mask] = change[mask] / volatility[mask]
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = (2.0 / (fast_period + 1)) ** 2
    slow_sc = (2.0 / (slow_period + 1)) ** 2
    
    # Initialize
    kama[period-1] = close[period-1]
    
    for i in range(period, n):
        if np.isnan(er[i]):
            kama[i] = kama[i-1]
        else:
            sc = er[i] * (fast_sc - slow_sc) + slow_sc
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
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

def calculate_roc(close, period=10):
    """Rate of Change - momentum indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i-period] > 1e-10:
            roc[i] = 100.0 * (close[i] - close[i-period]) / close[i-period]
    
    return roc

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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 6h indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    kama_6h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    roc_10 = calculate_roc(close, period=10)
    rsi_14 = calculate_rsi(close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    
    # Volatility Ratio for regime detection
    vol_ratio = np.full(n, np.nan, dtype=np.float64)
    mask = atr_30 > 1e-10
    vol_ratio[mask] = atr_7[mask] / atr_30[mask]
    
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
    min_bars = 60
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(kama_6h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Volatility Ratio) ===
        vol_rat = vol_ratio[i]
        is_vol_spike = vol_rat > 2.0  # Mean reversion regime
        is_calm_trend = vol_rat < 1.2  # Trend following regime
        
        # === TREND DIRECTION (1d KAMA bias) ===
        price_above_1d = close[i] > kama_1d_aligned[i]
        price_below_1d = close[i] < kama_1d_aligned[i]
        
        # === 6h KAMA TREND ===
        kama_slope_bull = kama_6h[i] > kama_6h[i-5] if i >= 5 and not np.isnan(kama_6h[i-5]) else False
        kama_slope_bear = kama_6h[i] < kama_6h[i-5] if i >= 5 and not np.isnan(kama_6h[i-5]) else False
        
        # === ROC MOMENTUM (LOOSE thresholds) ===
        roc_val = roc_10[i]
        roc_bullish = roc_val > -5.0  # Very loose - just not crashing
        roc_bearish = roc_val < 5.0   # Very loose - just not rallying hard
        roc_strong_bull = roc_val > 5.0
        roc_strong_bear = roc_val < -5.0
        
        # === RSI CONFIRMATION (LOOSE) ===
        rsi_val = rsi_14[i]
        rsi_not_overbought = rsi_val < 65
        rsi_not_oversold = rsi_val > 35
        rsi_oversold = rsi_val < 40
        rsi_overbought = rsi_val > 60
        
        # === BOLLINGER BAND POSITION ===
        bb_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) if (bb_upper[i] - bb_lower[i]) > 1e-10 else 0.5
        bb_touch_lower = close[i] <= bb_lower[i] * 1.01
        bb_touch_upper = close[i] >= bb_upper[i] * 0.99
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # CALM TREND REGIME: Trend following with KAMA + 1d bias + ROC
        if is_calm_trend:
            # LONG: 1d bullish + 6h KAMA up + ROC not negative + RSI ok
            if price_above_1d and kama_slope_bull and roc_bullish and rsi_not_overbought:
                desired_signal = SIZE_STRONG if roc_strong_bull else SIZE_BASE
            
            # SHORT: 1d bearish + 6h KAMA down + ROC not positive + RSI ok
            elif price_below_1d and kama_slope_bear and roc_bearish and rsi_not_oversold:
                desired_signal = -SIZE_STRONG if roc_strong_bear else -SIZE_BASE
        
        # VOL SPIKE REGIME: Mean reversion at BB extremes
        elif is_vol_spike:
            # LONG: Price at BB lower + RSI oversold + 1d not strongly bearish
            if bb_touch_lower and rsi_oversold:
                desired_signal = SIZE_BASE
            
            # SHORT: Price at BB upper + RSI overbought + 1d not strongly bullish
            elif bb_touch_upper and rsi_overbought:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Simple 1d bias + RSI range (MOST TRADES)
        else:
            # LONG: 1d bullish + RSI in neutral zone
            if price_above_1d and rsi_val > 40 and rsi_val < 60:
                desired_signal = SIZE_BASE
            
            # SHORT: 1d bearish + RSI in neutral zone
            elif price_below_1d and rsi_val > 40 and rsi_val < 60:
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