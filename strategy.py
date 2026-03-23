#!/usr/bin/env python3
"""
Experiment #1353: 1d Primary + 1w HTF — KAMA Adaptive Trend + Donchian Breakout

Hypothesis: Current best (mtf_1d_donchian_hma_rsi_1w_atr_v1, Sharpe=0.618) uses HMA which 
can whipsaw in range markets. KAMA (Kaufman Adaptive Moving Average) adapts smoothing 
based on market efficiency - faster in trends, slower in chop. This should reduce 
whipsaws while maintaining trend capture. Combined with 1w HMA macro filter and 
Donchian breakout trigger for entry timing.

Key design choices:
1. KAMA(10,2,30) on 1d - adapts to volatility, reduces chop whipsaws
2. 1w HMA(21) for macro trend bias - soft filter only
3. Donchian(20) breakout as entry trigger - captures momentum
4. RSI(14) with asymmetric thresholds (45/55) - confirms without over-filtering
5. ATR(14) trailing stop 3.0x - wider stop for 1d volatility
6. Position size 0.25 - conservative for daily swings
7. Multiple entry paths to ensure trade frequency (avoid 0-trade failure)

Target: 25-50 trades/year, Sharpe > 0.618, trades >= 30 train, >= 5 test
Timeframe: 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_donchian_rsi_1w_atr_adaptive_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """
    Kaufman Adaptive Moving Average - adapts smoothing based on market efficiency.
    ER (Efficiency Ratio) = net change / sum of absolute changes
    High ER = trending (use fast SC), Low ER = choppy (use slow SC)
    """
    n = len(close)
    if n < er_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    kama[er_period - 1] = close[er_period - 1]
    
    for i in range(er_period, n):
        # Efficiency Ratio: net price change / total volatility
        net_change = abs(close[i] - close[i - er_period])
        total_change = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        
        if total_change > 1e-10:
            er = net_change / total_change
        else:
            er = 0.0
        
        # Smoothing constant: scales between fast and slow based on ER
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index - asymmetric thresholds for entries"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss and volatility scaling"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels for entry triggers"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    kama = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
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
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(kama[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (1w HMA) - soft filter only ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (KAMA slope) ===
        kama_slope_bull = kama[i] > kama[i - 5] if i >= 5 else False
        kama_slope_bear = kama[i] < kama[i - 5] if i >= 5 else False
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # === RSI MOMENTUM (asymmetric thresholds for more trades) ===
        rsi_bull = rsi[i] > 45.0
        rsi_bear = rsi[i] < 55.0
        rsi_strong_bull = rsi[i] > 55.0
        rsi_strong_bear = rsi[i] < 45.0
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === DESIRED SIGNAL (multiple paths to ensure trade frequency) ===
        desired_signal = 0.0
        
        # LONG ENTRY PATHS
        # Path 1: Donchian breakout + KAMA slope + macro bull (primary entry)
        if breakout_long and kama_slope_bull and macro_bull:
            desired_signal = BASE_SIZE
        # Path 2: Price above KAMA + KAMA rising + RSI confirmation
        elif price_above_kama and kama_slope_bull and rsi_bull:
            desired_signal = BASE_SIZE * 0.6
        # Path 3: Strong momentum - price above both KAMA and 1w HMA
        elif price_above_kama and macro_bull and rsi_strong_bull:
            desired_signal = BASE_SIZE * 0.6
        
        # SHORT ENTRY PATHS
        # Path 1: Donchian breakout + KAMA slope + macro bear (primary entry)
        elif breakout_short and kama_slope_bear and macro_bear:
            desired_signal = -BASE_SIZE
        # Path 2: Price below KAMA + KAMA falling + RSI confirmation
        elif price_below_kama and kama_slope_bear and rsi_bear:
            desired_signal = -BASE_SIZE * 0.6
        # Path 3: Strong momentum - price below both KAMA and 1w HMA
        elif price_below_kama and macro_bear and rsi_strong_bear:
            desired_signal = -BASE_SIZE * 0.6
        
        # === STOPLOSS CHECK (Trailing ATR 3.0x for 1d volatility) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.1:
            final_signal = BASE_SIZE
        elif desired_signal < -0.1:
            final_signal = -BASE_SIZE
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