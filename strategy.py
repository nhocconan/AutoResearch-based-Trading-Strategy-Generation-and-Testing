#!/usr/bin/env python3
"""
Experiment #793: 1d Primary + 1w HTF — KAMA Adaptive Trend + Fisher Transform + Choppiness

Hypothesis: After 500+ failed strategies, key insights:
1. KAMA (Kaufman Adaptive MA) outperforms EMA/HMA in choppy crypto markets — adapts to volatility
2. Fisher Transform (Ehlers) catches reversals better than RSI/CRSI in bear markets
3. 1d timeframe needs SIMPLE entry logic to generate trades (complex = 0 trades)
4. Choppiness Index for regime is good but thresholds need adjustment (50/50 split)
5. 1w KAMA provides strong trend bias without lag
6. Relaxed entry conditions: Fisher > -1.5 (not -0.5) for more longs

Strategy design:
1. 1w KAMA(21) for long-term trend bias (aligned via mtf_data helper)
2. 1d KAMA(21) for adaptive trend following
3. 1d Fisher Transform(9) for entry timing
4. 1d Choppiness Index(14) for regime detection
5. 1d ATR(14) for trailing stop (2.5x)
6. Discrete signals: 0.0, ±0.25, ±0.30
7. SIMPLE entry logic to ensure >=10 trades/train, >=3 trades/test

Key differences from failed strategies:
- KAMA instead of HMA/EMA — adapts to volatility automatically
- Fisher Transform instead of CRSI — better reversal detection
- Simpler regime: CHOP > 50 = range, < 50 = trend (not 61.8/38.2)
- Fisher thresholds: -1.5/+1.5 (not -0.5/+0.5) — more signals
- Hold logic: maintain position until Fisher reverses OR stoploss

Target: Sharpe > 0.612, trades >= 10 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 15-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_fisher_chop_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average.
    Adapts to market volatility — fast in trends, slow in chop.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        change = np.abs(close[i] - close[i - period])
        if change == 0:
            er[i] = 0
            continue
        volatility = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if volatility == 0:
            er[i] = 0
        else:
            er[i] = change / volatility
    
    er = np.clip(er, 0, 1)
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    sc = np.clip(sc, slow_sc, fast_sc)
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        if np.isnan(kama[i-1]):
            continue
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform.
    Converts price to Gaussian distribution for better reversal detection.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_prev
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i-period+1:i+1] + low[i-period+1:i+1]) / 2
        highest_hl2 = np.max(hl2)
        lowest_hl2 = np.min(hl2)
        
        if highest_hl2 == lowest_hl2:
            fisher[i] = 0
            fisher_prev[i] = fisher[i-1] if i > period else 0
            continue
        
        # Normalize to -1 to +1
        x = (2 * hl2[-1] - highest_hl2 - lowest_hl2) / (highest_hl2 - lowest_hl2 + 1e-10)
        x = np.clip(x, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x) + 1e-10)
        fisher[i] = np.clip(fisher[i], -5, 5)
        
        if i > period:
            fisher_prev[i] = fisher[i-1]
        else:
            fisher_prev[i] = 0
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 50 = ranging, CHOP < 50 = trending (simplified).
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            prev_close = close[j-1] if j > 0 else close[j]
            tr_sum += max(high[j] - low[j], np.abs(high[j] - prev_close), np.abs(low[j] - prev_close))
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        prev_close = close[i-1]
        tr[i] = max(high[i] - low[i], np.abs(high[i] - prev_close), np.abs(low[i] - prev_close))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    kama_1d = calculate_kama(close, period=21)
    fisher_1d, fisher_prev_1d = calculate_fisher_transform(high, low, period=9)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align HTF KAMA for trend bias
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=21)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(kama_1d[i]) or np.isnan(fisher_1d[i]) or np.isnan(atr_1d[i]):
            continue
        if atr_1d[i] <= 1e-10:
            continue
        if np.isnan(kama_1w_aligned[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(chop_1d[i]):
            continue
        
        # === TREND BIAS (1w HTF KAMA21) ===
        trend_1w_bullish = close[i] > kama_1w_aligned[i]
        trend_1w_bearish = close[i] < kama_1w_aligned[i]
        
        # === 1D TREND (KAMA slope) ===
        kama_slope_bullish = kama_1d[i] > kama_1d[i-5] if not np.isnan(kama_1d[i-5]) else False
        kama_slope_bearish = kama_1d[i] < kama_1d[i-5] if not np.isnan(kama_1d[i-5]) else False
        
        # === REGIME DETECTION (Choppiness Index) ===
        ranging_regime = chop_1d[i] > 50
        trending_regime = chop_1d[i] < 50
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher_1d[i] < -1.5
        fisher_overbought = fisher_1d[i] > 1.5
        fisher_cross_up = fisher_prev_1d[i] < -1.5 and fisher_1d[i] >= -1.5
        fisher_cross_down = fisher_prev_1d[i] > 1.5 and fisher_1d[i] <= 1.5
        fisher_rising = fisher_1d[i] > fisher_prev_1d[i] if not np.isnan(fisher_prev_1d[i]) else False
        fisher_falling = fisher_1d[i] < fisher_prev_1d[i] if not np.isnan(fisher_prev_1d[i]) else False
        
        # === PRICE POSITION ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        price_above_kama = close[i] > kama_1d[i]
        price_below_kama = close[i] < kama_1d[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRY LOGIC ===
        long_conditions = 0
        
        # Condition 1: Fisher oversold + trend alignment
        if fisher_oversold and trend_1w_bullish:
            long_conditions += 1
        
        # Condition 2: Fisher cross up + 1d KAMA bullish
        if fisher_cross_up and kama_slope_bullish:
            long_conditions += 1
        
        # Condition 3: Price above KAMA + Fisher rising + 1w bullish
        if price_above_kama and fisher_rising and trend_1w_bullish:
            long_conditions += 1
        
        # Condition 4: Range regime + Fisher oversold (mean reversion)
        if ranging_regime and fisher_oversold:
            long_conditions += 1
        
        # Condition 5: Trend regime + pullback to KAMA
        if trending_regime and trend_1w_bullish and price_below_kama and fisher_rising:
            long_conditions += 1
        
        if long_conditions >= 1:
            desired_signal = BASE_SIZE if long_conditions >= 2 else REDUCED_SIZE
        
        # === SHORT ENTRY LOGIC ===
        short_conditions = 0
        
        # Condition 1: Fisher overbought + trend alignment
        if fisher_overbought and trend_1w_bearish:
            short_conditions += 1
        
        # Condition 2: Fisher cross down + 1d KAMA bearish
        if fisher_cross_down and kama_slope_bearish:
            short_conditions += 1
        
        # Condition 3: Price below KAMA + Fisher falling + 1w bearish
        if price_below_kama and fisher_falling and trend_1w_bearish:
            short_conditions += 1
        
        # Condition 4: Range regime + Fisher overbought (mean reversion)
        if ranging_regime and fisher_overbought:
            short_conditions += 1
        
        # Condition 5: Trend regime + rally to KAMA
        if trending_regime and trend_1w_bearish and price_above_kama and fisher_falling:
            short_conditions += 1
        
        if short_conditions >= 1:
            if desired_signal > 0:
                # Conflict — stay flat or reduce
                desired_signal = 0.0
            else:
                desired_signal = -BASE_SIZE if short_conditions >= 2 else -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if Fisher not overbought and 1w trend intact
                if fisher_1d[i] < 1.0 and trend_1w_bullish:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if Fisher not oversold and 1w trend intact
                if fisher_1d[i] > -1.0 and trend_1w_bearish:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if Fisher overbought or 1w trend reverses
            if fisher_overbought or (trend_1w_bearish and fisher_falling):
                desired_signal = 0.0
            # Exit if price far below KAMA in trending regime
            if trending_regime and close[i] < kama_1d[i] * 0.95:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if Fisher oversold or 1w trend reverses
            if fisher_oversold or (trend_1w_bullish and fisher_rising):
                desired_signal = 0.0
            # Exit if price far above KAMA in trending regime
            if trending_regime and close[i] > kama_1d[i] * 1.05:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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
        
        signals[i] = desired_signal
    
    return signals