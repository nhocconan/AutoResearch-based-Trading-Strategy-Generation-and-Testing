#!/usr/bin/env python3
"""
Hypothesis: 30m primary with 4h HMA trend filter + KAMA adaptive trend following
- 4h HMA(21/50) provides stable trend bias (bull/bear/neutral)
- 30m KAMA(10,2,30) adapts to volatility - faster in trends, slower in chop
- KAMA crossover signals entries in direction of HTF trend
- ATR(14) trailing stop for risk management
- Simpler entry logic to ensure sufficient trades (learned from 30m failures)
- Discrete position sizing: 0.0, ±0.20, ±0.30 to minimize fee churn
Timeframe: 30m (primary), 4h (HTF trend filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_hma_trend_30m_v1"
timeframe = "30m"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    kama = np.zeros(n)
    
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.abs(close[:er_period] - close[0])
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    volatility[:er_period] = change[:er_period]
    
    er = np.zeros(n)
    er[er_period:] = change[er_period:] / (volatility[er_period:] + 1e-10)
    er[:er_period] = 1.0
    
    # Smoothing Constant
    sc = (er * (fast_sc - slow_sc) / (fast_sc + slow_sc) + slow_sc / (fast_sc + slow_sc)) ** 2
    
    # KAMA calculation
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    hull = 2 * wma_half - wma_full
    hma = pd.Series(hull).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # 4h HMA for trend direction
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, 50)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    
    # 30m indicators
    kama_30m = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    kama_fast = calculate_kama(close, er_period=5, fast_sc=2, slow_sc=20)
    atr = calculate_atr(high, low, close, period=14)
    rsi_30m = calculate_rsi(close, 14)
    
    # Price momentum
    mom_5 = close / np.roll(close, 5) - 1
    mom_5[:5] = 0.0
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30   # 30% for longs in bull trend
    SIZE_SHORT = 0.20  # 20% for shorts (asymmetric - bear trends are sharper)
    SIZE_NEUTRAL = 0.15  # Reduced size in neutral
    
    prev_signal = 0.0
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    last_entry_idx = -100
    
    for i in range(100, n):
        # HTF trend regime (4h HMA crossover)
        htf_bull = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        htf_bear = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # 4h trend strength (slope)
        if i >= 8:
            hma_slope = (hma_4h_21_aligned[i] - hma_4h_21_aligned[i-8]) / (hma_4h_21_aligned[i-8] + 1e-10)
        else:
            hma_slope = 0.0
        trend_strength = min(abs(hma_slope) * 50, 1.5)
        
        # 30m KAMA crossover signals
        kama_cross_long = kama_fast[i] > kama_30m[i] and kama_fast[i-1] <= kama_30m[i-1]
        kama_cross_short = kama_fast[i] < kama_30m[i] and kama_fast[i-1] >= kama_30m[i-1]
        
        # KAMA position (above/below)
        kama_above = kama_fast[i] > kama_30m[i]
        kama_below = kama_fast[i] < kama_30m[i]
        
        # Price vs KAMA
        price_above_kama = close[i] > kama_30m[i]
        price_below_kama = close[i] < kama_30m[i]
        
        # RSI filter (avoid extreme entries)
        rsi_ok_long = rsi_30m[i] < 70
        rsi_ok_short = rsi_30m[i] > 30
        
        # Momentum confirmation
        mom_positive = mom_5[i] > 0.005
        mom_negative = mom_5[i] < -0.005
        
        # ATR stoploss level
        atr_stop_mult = 2.5
        atr_stop = atr_stop_mult * atr[i]
        
        # Check stoploss first (CRITICAL - Rule 6)
        if position_side == 1 and i - last_entry_idx > 5:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - atr_stop
            if close[i] < trailing_stop or close[i] < entry_price - atr_stop:
                signals[i] = 0.0
                position_side = 0
                prev_signal = 0.0
                continue
        elif position_side == -1 and i - last_entry_idx > 5:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + atr_stop
            if close[i] > trailing_stop or close[i] > entry_price + atr_stop:
                signals[i] = 0.0
                position_side = 0
                prev_signal = 0.0
                continue
        
        # Determine base size based on HTF regime
        if htf_bull:
            base_size = SIZE_LONG
        elif htf_bear:
            base_size = SIZE_SHORT
        else:
            base_size = SIZE_NEUTRAL
        
        # Scale size by trend strength
        size = base_size * (0.7 + 0.3 * trend_strength)
        size = min(max(size, 0.15), 0.35)  # Clamp to valid range
        
        # Entry logic - simpler to ensure trades are generated
        if htf_bull:  # Bull regime - prefer longs
            # KAMA crossover long entry
            if kama_cross_long and rsi_ok_long and price_above_kama:
                signals[i] = size
                if prev_signal <= 0:
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
                    last_entry_idx = i
            # KAMA above + momentum continuation
            elif kama_above and price_above_kama and mom_positive and prev_signal <= 0:
                signals[i] = size * 0.8
                position_side = 1
                entry_price = close[i]
                highest_since_entry = close[i]
                last_entry_idx = i
            # RSI oversold in uptrend - buy dip
            elif rsi_30m[i] < 40 and kama_above and htf_bull:
                signals[i] = size
                if prev_signal <= 0:
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
                    last_entry_idx = i
            # RSI overbought - reduce or exit
            elif rsi_30m[i] > 75 and position_side == 1:
                signals[i] = size * 0.3
            else:
                signals[i] = prev_signal if position_side == 1 else 0.0
                
        elif htf_bear:  # Bear regime - prefer shorts
            # KAMA crossover short entry
            if kama_cross_short and rsi_ok_short and price_below_kama:
                signals[i] = -size
                if prev_signal >= 0:
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
                    last_entry_idx = i
            # KAMA below + momentum continuation
            elif kama_below and price_below_kama and mom_negative and prev_signal >= 0:
                signals[i] = -size * 0.8
                position_side = -1
                entry_price = close[i]
                lowest_since_entry = close[i]
                last_entry_idx = i
            # RSI overbought in downtrend - sell rip
            elif rsi_30m[i] > 60 and kama_below and htf_bear:
                signals[i] = -size
                if prev_signal >= 0:
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
                    last_entry_idx = i
            # RSI oversold - reduce or exit
            elif rsi_30m[i] < 25 and position_side == -1:
                signals[i] = -size * 0.3
            else:
                signals[i] = prev_signal if position_side == -1 else 0.0
                
        else:  # Neutral regime - smaller positions, mean reversion
            if kama_cross_long and rsi_30m[i] < 50:
                signals[i] = SIZE_NEUTRAL
                if prev_signal <= 0:
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
                    last_entry_idx = i
            elif kama_cross_short and rsi_30m[i] > 50:
                signals[i] = -SIZE_NEUTRAL
                if prev_signal >= 0:
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
                    last_entry_idx = i
            elif abs(rsi_30m[i] - 50) < 10:
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = prev_signal
        
        # Discretize signal to reduce churn (CRITICAL - Rule 4)
        if abs(signals[i]) < 0.10:
            signals[i] = 0.0
        elif signals[i] > 0:
            # Round to discrete levels: 0.15, 0.20, 0.25, 0.30, 0.35
            signals[i] = round(signals[i] / 0.05) * 0.05
            signals[i] = min(max(signals[i], 0.15), 0.35)
        else:
            signals[i] = round(signals[i] / 0.05) * 0.05
            signals[i] = max(min(signals[i], -0.15), -0.35)
        
        prev_signal = signals[i]
    
    return signals