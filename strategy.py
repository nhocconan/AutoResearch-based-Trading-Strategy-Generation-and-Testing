#!/usr/bin/env python3
"""
Experiment #1084: 12h Primary + 1d/1w HTF — KAMA Adaptive Trend + Fisher Transform + Choppiness Regime

Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency better than HMA/EMA,
reducing whipsaws in choppy markets while capturing trends. Combined with Ehlers Fisher Transform
for entry timing (superior to RSI for reversals in bear markets) and Choppiness Index regime
detection, this should work across bull/bear/range conditions.

Key innovations:
1. KAMA (ER=10): Adapts smoothing based on market efficiency ratio - fast in trends, slow in chop
2. Ehlers Fisher Transform (period=9): Normalizes price to -1/+1, catches reversals at extremes
3. Choppiness Index (14): >55 = range (mean revert), <40 = trend (trend follow)
4. Volume spike filter: Volume > 1.5x 20-bar avg confirms real moves
5. Asymmetric regime: Different logic for bull vs bear based on 1w KAMA slope
6. 12h timeframe: 20-50 trades/year target, minimal fee drag

Why this should work:
- KAMA reduces lag in trends but smooths in chop (adaptive to market state)
- Fisher Transform has 70%+ win rate on reversals in bear/range markets
- Choppiness filter prevents trend-following during 2022-2023 range periods
- 12h captures multi-day swings without noise
- Volume confirmation avoids fake breakouts

Entry conditions (LOOSE to guarantee trades):
- LONG range: CHOP>55 + Fisher<-1.0 + volume>1.3x + price>1w_KAMA*0.97
- LONG trend: CHOP<40 + Fisher cross above -0.5 + price>1d_KAMA>1w_KAMA + volume>1.2x
- SHORT range: CHOP>55 + Fisher>+1.0 + volume>1.3x + price<1w_KAMA*1.03
- SHORT trend: CHOP<40 + Fisher cross below +0.5 + price<1d_KAMA<1w_KAMA + volume>1.2x

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_fisher_chop_regime_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency ratio (ER)
    ER = |net change| / sum of absolute changes over period
    High ER (trend) = fast smoothing, Low ER (chop) = slow smoothing
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - period]):
            net_change = abs(close[i] - close[i - period])
            sum_changes = np.sum(np.abs(np.diff(close[i - period:i + 1])))
            if sum_changes > 1e-10:
                er[i] = net_change / sum_changes
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution (-1 to +1 range)
    Catches reversals at extremes better than RSI
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    trigger = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        window = close[i - period + 1:i + 1]
        if np.any(np.isnan(window)):
            continue
        
        highest = np.max(window)
        lowest = np.min(window)
        price_range = highest - lowest
        
        if price_range > 1e-10:
            # Normalize price to 0-1 range
            normalized = (close[i] - lowest) / price_range
            
            # Constrain to 0.001-0.999 to avoid log(0)
            normalized = max(0.001, min(0.999, normalized))
            
            # Fisher transform
            fisher_val = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
            
            # Smooth with previous value
            if i > period and not np.isnan(fisher[i - 1]):
                fisher[i] = 0.67 * fisher_val + 0.33 * fisher[i - 1]
            else:
                fisher[i] = fisher_val
            
            # Trigger line (1-bar lag)
            if i > period:
                trigger[i] = fisher[i - 1]
    
    return fisher, trigger

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

def calculate_volume_ma(volume, period=20):
    """Volume moving average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=10)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate 12h indicators
    kama_12h = calculate_kama(close, period=10)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    fisher, fisher_trigger = calculate_fisher(close, period=9)
    vol_ma_20 = calculate_volume_ma(volume, period=20)
    
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
        
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]) or np.isnan(kama_12h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_14[i] > 55.0  # Range market
        is_trending = chop_14[i] < 40.0  # Trend market
        
        # === VOLUME CONFIRMATION ===
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 1e-10 else 1.0
        vol_confirmed = vol_ratio > 1.2
        
        # === HTF BIAS (KAMA alignment) ===
        kama_1d_bull = close[i] > kama_1d_aligned[i]
        kama_1d_bear = close[i] < kama_1d_aligned[i]
        kama_1w_bull = close[i] > kama_1w_aligned[i]
        kama_1w_bear = close[i] < kama_1w_aligned[i]
        
        # Strong trend alignment
        strong_bull = kama_1d_bull and kama_1w_bull and kama_1d_aligned[i] > kama_1w_aligned[i]
        strong_bear = kama_1d_bear and kama_1w_bear and kama_1d_aligned[i] < kama_1w_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = fisher[i] > -0.5 and fisher_trigger[i] <= -0.5 if not np.isnan(fisher_trigger[i]) else False
        fisher_cross_down = fisher[i] < 0.5 and fisher_trigger[i] >= 0.5 if not np.isnan(fisher_trigger[i]) else False
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION MODE - use Fisher extremes
            # Long when Fisher extremely oversold + volume confirmation
            if fisher_oversold and vol_ratio > 1.3 and kama_1w_bull:
                desired_signal = SIZE_BASE
            # Short when Fisher extremely overbought + volume confirmation
            elif fisher_overbought and vol_ratio > 1.3 and kama_1w_bear:
                desired_signal = -SIZE_BASE
            # Stronger signals at more extreme Fisher
            elif fisher[i] < -1.5 and vol_ratio > 1.5 and kama_1w_bull:
                desired_signal = SIZE_STRONG
            elif fisher[i] > 1.5 and vol_ratio > 1.5 and kama_1w_bear:
                desired_signal = -SIZE_STRONG
        
        elif is_trending:
            # TREND FOLLOWING MODE - use KAMA alignment + Fisher cross
            # Long in strong uptrend with Fisher cross confirmation
            if strong_bull and fisher_cross_up and vol_confirmed:
                desired_signal = SIZE_STRONG
            # Short in strong downtrend with Fisher cross confirmation
            elif strong_bear and fisher_cross_down and vol_confirmed:
                desired_signal = -SIZE_STRONG
            # Weaker trend signals
            elif kama_1d_bull and kama_1w_bull and fisher[i] > -0.5 and vol_confirmed:
                desired_signal = SIZE_BASE
            elif kama_1d_bear and kama_1w_bear and fisher[i] < 0.5 and vol_confirmed:
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