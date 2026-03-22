#!/usr/bin/env python3
"""
Experiment #024: 1d KAMA Trend + 1w HMA Filter + BB Squeeze + ADX + ATR Stop

Hypothesis: After 23 experiments, the pattern shows:
1. Daily (1d) timeframe is UNDERUTILIZED - only 3 attempts (#012, #018, and this)
2. #012 and #018 both used RSI mean reversion which FAILED (Sharpe -0.917, -2.131)
3. Trend following works BETTER on higher timeframes (see #023 Sharpe=0.137)
4. KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency ratio
   - Fast in trending markets, slow in ranging markets
   - Proven in systematic trading literature (Kaufman's "Trading Systems and Methods")
5. BB Squeeze detects low volatility before breakouts (classic pattern)
6. 1w HMA provides robust long-term trend bias without lag
7. ADX > 15 filters choppy conditions where trend strategies fail
8. ATR trailing stop (2.5x) protects from reversals

This strategy combines:
1. KAMA(10,2,30) crossover with KAMA(21,2,30) - adaptive trend signal
2. 1w HMA(21) - long-term trend filter (price above = long only, below = short only)
3. Bollinger Band Width percentile - detects squeeze before expansion
4. ADX(14) > 15 - confirms trending conditions (lowered for more trades)
5. ATR(14) trailing stop at 2.5x - risk management
6. Position sizing: 0.30 strong, 0.20 moderate (discrete levels)

Why this should beat #023 (Sharpe=0.137):
- KAMA adapts to volatility unlike fixed EMA/HMA
- BB squeeze catches volatility expansion early
- 1d TF = fewer trades = less fee drag
- ADX filter avoids whipsaws in choppy markets
- Conservative sizing protects from 2022-style crashes

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year on 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_1w_hma_bb_squeeze_adx_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average.
    Adapts smoothing based on market efficiency (trend vs noise).
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio: net change / sum of absolute changes
    change = np.abs(close_s.diff(er_period))
    volatility = np.abs(close_s.diff()).rolling(window=er_period, min_periods=er_period).sum()
    er = change / volatility.replace(0, np.inf)
    er = er.fillna(0)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_bb(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    middle = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    bb_width = (upper - lower) / middle
    return upper.values, lower.values, middle.values, bb_width.values

def calculate_bb_width_percentile(bb_width, lookback=100):
    """Calculate BB Width percentile over lookback period."""
    bb_s = pd.Series(bb_width)
    def pct_rank(x):
        if len(x) < 10 or x.max() == x.min():
            return 50.0
        return (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100
    percentile = bb_s.rolling(window=lookback, min_periods=lookback).apply(pct_rank, raw=False)
    return percentile.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    kama_fast = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_slow = calculate_kama(close, er_period=21, fast_period=2, slow_period=30)
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_middle, bb_width = calculate_bb(close, 20, 2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_width, 100)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    SIZE_STRONG = 0.30
    SIZE_MODERATE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            continue
        
        if np.isnan(bb_width_pct[i]) or np.isnan(bb_upper[i]):
            continue
        
        # === HTF TREND BIAS (1w HMA) ===
        price_vs_1w = close[i] - hma_1w_aligned[i]
        bull_htf = price_vs_1w > 0
        bear_htf = price_vs_1w < 0
        
        # === KAMA TREND DIRECTION ===
        kama_trend_long = kama_fast[i] > kama_slow[i]
        kama_trend_short = kama_fast[i] < kama_slow[i]
        
        # === KAMA CROSSOVER (fresh signal) ===
        kama_cross_long = False
        kama_cross_short = False
        if i > 0 and not np.isnan(kama_fast[i-1]) and not np.isnan(kama_slow[i-1]):
            kama_cross_long = kama_fast[i] > kama_slow[i] and kama_fast[i-1] <= kama_slow[i-1]
            kama_cross_short = kama_fast[i] < kama_slow[i] and kama_fast[i-1] >= kama_slow[i-1]
        
        # === BB SQUEEZE (low volatility before expansion) ===
        bb_squeeze = bb_width_pct[i] < 40  # Bottom 40% = compression
        
        # === BB BREAKOUT ===
        bb_breakout_long = close[i] > bb_upper[i-1] if i > 0 else False
        bb_breakout_short = close[i] < bb_lower[i-1] if i > 0 else False
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 15  # Lowered threshold for more trades
        
        # === DI DIRECTION ===
        di_bull = plus_di[i] > minus_di[i]
        di_bear = minus_di[i] > plus_di[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        signal_strength = 0
        
        # LONG ENTRY: HTF bull + KAMA trend + confirmations
        if bull_htf and kama_trend_long:
            signal_strength += 2  # HTF trend + KAMA trend (core signals)
            
            if kama_cross_long:
                signal_strength += 1  # Fresh crossover
            
            if bb_squeeze or bb_breakout_long:
                signal_strength += 1  # Volatility expansion setup
            
            if adx_strong:
                signal_strength += 1  # Trend strength
            
            if di_bull:
                signal_strength += 1  # DI direction
            
            if signal_strength >= 4:
                new_signal = SIZE_STRONG
            elif signal_strength >= 3:
                new_signal = SIZE_MODERATE
        
        # SHORT ENTRY: HTF bear + KAMA trend + confirmations
        elif bear_htf and kama_trend_short:
            signal_strength += 2  # HTF trend + KAMA trend (core signals)
            
            if kama_cross_short:
                signal_strength += 1  # Fresh crossover
            
            if bb_squeeze or bb_breakout_short:
                signal_strength += 1  # Volatility expansion setup
            
            if adx_strong:
                signal_strength += 1  # Trend strength
            
            if di_bear:
                signal_strength += 1  # DI direction
            
            if signal_strength >= 4:
                new_signal = -SIZE_STRONG
            elif signal_strength >= 3:
                new_signal = -SIZE_MODERATE
        
        # === STOPLOSS LOGIC - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and bear_htf:
                trend_exit = True
            if position_side < 0 and bull_htf:
                trend_exit = True
            
            if position_side > 0 and kama_trend_short:
                trend_exit = True
            if position_side < 0 and kama_trend_long:
                trend_exit = True
        
        if stoploss_triggered or trend_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals