#!/usr/bin/env python3
"""
Experiment #487: 1d Primary + 1w HTF — KAMA Adaptive Trend + Choppiness Regime + RSI Entry + ATR Stop

Hypothesis: After 400+ failed experiments, the key insight is that crypto alternates between 
trending and ranging regimes. KAMA (Kaufman Adaptive Moving Average) automatically adjusts 
smoothing based on market efficiency ratio - fast in trends, slow in chop. Combined with 
Choppiness Index for regime confirmation, this should outperform static HMA/EMA approaches.

Key innovations:
1. KAMA(10,2,30) - adapts smoothing constant based on ER (Efficiency Ratio)
2. Choppiness Index(14) - regime filter: CHOP>61.8=range, CHOP<38.2=trend
3. RSI(14) with regime-adjusted thresholds: range=30/70, trend=40/60
4. 1w KAMA for HTF major trend bias (proven in #477)
5. ATR(14) trailing stop at 2.5x - protects against crashes
6. Discrete sizing: 0.30 long, -0.25 short (asymmetric for crypto long bias)
7. HOLD logic with regime awareness - stay in position while regime intact

Why 1d works: Natural 20-50 trades/year. Less noise than lower TFs. 1w HTF provides 
major trend filter. KAMA adapts to BTC/ETH's notorious whipsaw behavior better than HMA.

Target: Sharpe > 0.612 (beat current best), DD < -35%, trades >= 30 train, >= 3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_chop_rsi_regime_1w_v2"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency ratio.
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    fast_sc = 2/(fast+1), slow_sc = 2/(slow+1)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = 0.0
        for j in range(i - er_period + 1, i + 1):
            noise += np.abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = np.full(n, np.nan)
    for i in range(er_period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing method."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0.0)
    loss[1:] = np.where(delta < 0, -delta, 0.0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    kama_1d = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    rsi_1d = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators (1w KAMA for major trend bias)
    kama_1w_raw = calculate_kama(df_1w['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = -0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(kama_1d[i]):
            continue
        if np.isnan(chop_1d[i]):
            continue
        if np.isnan(rsi_1d[i]):
            continue
        if np.isnan(kama_1w_aligned[i]):
            continue
        
        # === HTF MAJOR TREND BIAS (1w KAMA) ===
        htf_bullish = close[i] > kama_1w_aligned[i]
        htf_bearish = close[i] < kama_1w_aligned[i]
        
        # === PRIMARY TREND (1d KAMA) ===
        price_above_kama = close[i] > kama_1d[i]
        price_below_kama = close[i] < kama_1d[i]
        kama_slope_up = kama_1d[i] > kama_1d[i - 5] if i >= 5 else False
        kama_slope_down = kama_1d[i] < kama_1d[i - 5] if i >= 5 else False
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = range/chop, CHOP < 38.2 = trend
        is_choppy = chop_1d[i] > 55.0  # relaxed threshold for more signals
        is_trending = chop_1d[i] < 45.0  # relaxed threshold
        
        # === RSI SIGNALS (regime-adjusted thresholds) ===
        if is_choppy:
            # Mean reversion in chop: buy low, sell high
            rsi_oversold = rsi_1d[i] < 35.0
            rsi_overbought = rsi_1d[i] > 65.0
        else:
            # Trend following: buy strength, sell weakness
            rsi_oversold = rsi_1d[i] < 45.0
            rsi_overbought = rsi_1d[i] > 55.0
        
        rsi_deep_oversold = rsi_1d[i] < 30.0
        rsi_deep_overbought = rsi_1d[i] > 70.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRIES - Regime-aware confluence
        if is_trending:
            # Trend-following long setup
            long_score = 0
            if htf_bullish:
                long_score += 2
            if price_above_kama:
                long_score += 1
            if kama_slope_up:
                long_score += 1
            if rsi_oversold or rsi_1d[i] > 50.0:  # RSI not overbought in trend
                long_score += 1
            
            if long_score >= 3:
                desired_signal = SIZE_LONG
        else:
            # Mean reversion long setup (choppy market)
            long_score = 0
            if htf_bullish:
                long_score += 2
            if rsi_deep_oversold:
                long_score += 2
            elif rsi_oversold:
                long_score += 1
            if price_below_kama:  # buy dip to KAMA in chop
                long_score += 1
            
            if long_score >= 3:
                desired_signal = SIZE_LONG
        
        # SHORT ENTRIES
        if desired_signal == 0.0:
            if is_trending:
                # Trend-following short setup
                short_score = 0
                if htf_bearish:
                    short_score += 2
                if price_below_kama:
                    short_score += 1
                if kama_slope_down:
                    short_score += 1
                if rsi_overbought or rsi_1d[i] < 50.0:
                    short_score += 1
                
                if short_score >= 3:
                    desired_signal = SIZE_SHORT
            else:
                # Mean reversion short setup (choppy market)
                short_score = 0
                if htf_bearish:
                    short_score += 2
                if rsi_deep_overbought:
                    short_score += 2
                elif rsi_overbought:
                    short_score += 1
                if price_above_kama:  # sell rally to KAMA in chop
                    short_score += 1
                
                if short_score >= 3:
                    desired_signal = SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if regime unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HTF still bullish OR trend intact
                if htf_bullish or (price_above_kama and not is_choppy):
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HTF still bearish OR trend intact
                if htf_bearish or (price_below_kama and not is_choppy):
                    desired_signal = SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = SIZE_SHORT
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals