#!/usr/bin/env python3
"""
Experiment #1423: 1d Primary + 1w HTF — Fisher Transform + KAMA Adaptive Trend

Hypothesis: The Ehlers Fisher Transform excels at catching reversals in bear/range markets
(2025 test period) by normalizing price into a bounded Gaussian distribution. Combined with
KAMA (Kaufman Adaptive Moving Average) which adapts to market efficiency ratio, this should
outperform static EMA/HMA in volatile regimes.

Why this differs from failed experiments:
- #1417 used Choppiness+CRSI+Donchian (Sharpe=-0.416) — too many conflicting filters
- #1421 used regime-adaptive CRSI (Sharpe=-0.481) — regime detection added complexity without edge
- This uses Fisher Transform (proven in research for bear market reversals) + KAMA (adaptive trend)
- Simpler entry logic: Fisher extreme + KAMA slope + 1w HMA trend filter
- Target: 25-45 trades/year (fewer than 4h strategies, more sustainable fee drag)

Key innovations:
1. Fisher Transform period=9, entry when Fisher crosses -1.5 (long) or +1.5 (short)
2. KAMA ER-based adaptation — faster in trends, slower in chop
3. 1w HMA as single macro filter (not multiple HTF conflicting signals)
4. ATR trailing stop 2.5x for risk management
5. Position size 0.30 (conservative for daily volatility)

Target: Sharpe > 0.618 (beat current best), trades >= 30 train, >= 5 test, DD > -50%
Timeframe: 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_kama_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price into bounded Gaussian distribution.
    Excellent for identifying reversal points in bear/range markets.
    Reference: Ehlers, J.F. "The Fisher Transform" Technical Analysis of Stocks & Commodities, 2002.
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)  # Previous bar's fisher value for crossover detection
    
    # Calculate typical price and normalize
    for i in range(period - 1, n):
        # Find highest high and lowest low over lookback
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            # Normalize price to -1 to +1 range
            price_range = highest_high - lowest_low
            normalized = 2.0 * ((high[i] + low[i]) / 2.0 - lowest_low) / price_range - 1.0
            
            # Clamp to avoid extreme values
            normalized = max(-0.999, min(0.999, normalized))
            
            # Apply Fisher transform
            fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
            
            # Store previous value for crossover detection
            if i > period - 1:
                fisher_signal[i] = fisher[i-1]
            else:
                fisher_signal[i] = 0.0
    
    return fisher, fisher_signal

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average - adapts smoothing based on market efficiency.
    Fast SC in trending markets, slow SC in choppy markets.
    Reference: Kaufman, P.J. "Trading Systems and Methods", 5th Edition.
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i-er_period])
        if price_change > 1e-10:
            volatility = np.nansum(np.abs(np.diff(close[i-er_period:i+1])))
            er[i] = price_change / volatility
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant (SC)
    # SC = ER * (fast_SC - slow_SC) + slow_SC
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA with SMA
    kama[er_period] = np.nanmean(close[:er_period+1])
    
    # Calculate adaptive KAMA
    for i in range(er_period + 1, n):
        if not np.isnan(er[i]) and not np.isnan(kama[i-1]):
            sc = er[i] * (fast_sc - slow_sc) + slow_sc
            kama[i] = kama[i-1] + sc * sc * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA, less lag"""
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
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss sizing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index - additional filter for entry confirmation"""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
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
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1w HMA) - primary filter ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === KAMA TREND DIRECTION ===
        kama_bull = kama[i] > kama[i-1] if i > 0 else False
        kama_bear = kama[i] < kama[i-1] if i > 0 else False
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        fisher_long_cross = fisher_signal[i] < -1.5 and fisher[i] >= -1.5
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_short_cross = fisher_signal[i] > 1.5 and fisher[i] <= 1.5
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi[i] < 45.0  # More lenient for more trades
        rsi_overbought = rsi[i] > 55.0  # More lenient for more trades
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: Fisher reversal + KAMA bullish + macro bull + RSI confirmation
        if fisher_long_cross and kama_bull and macro_bull and rsi_oversold:
            desired_signal = BASE_SIZE
        # Alternative long: Strong Fisher reversal even without KAMA confirmation
        elif fisher[i] < -1.8 and macro_bull and rsi[i] < 40.0:
            desired_signal = BASE_SIZE * 0.7
        
        # SHORT ENTRY: Fisher reversal + KAMA bearish + macro bear + RSI confirmation
        elif fisher_short_cross and kama_bear and macro_bear and rsi_overbought:
            desired_signal = -BASE_SIZE
        # Alternative short: Strong Fisher reversal even without KAMA confirmation
        elif fisher[i] > 1.8 and macro_bear and rsi[i] > 60.0:
            desired_signal = -BASE_SIZE * 0.7
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE if desired_signal > 0 else -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
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
                # Flip position
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