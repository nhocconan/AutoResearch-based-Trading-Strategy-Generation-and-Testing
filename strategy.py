#!/usr/bin/env python3
"""
Experiment #1521: 4h Primary + 1d/1w HTF — KAMA Adaptive Trend + ADX Strength + RSI Pullback

Hypothesis: Based on proven patterns (KAMA+ADX+Chop ETH Sharpe +0.755, HMA+RSI SOL +0.879),
combining adaptive trend (KAMA) with trend strength (ADX) and pullback entries (RSI) should work.
Key insights from 1100+ failed strategies:
1. KAMA adapts to volatility better than HMA/EMA in choppy markets (2022 crash, 2025 range)
2. ADX>20 (not >30) filters weak trends without killing trade count
3. 1d HTF trend bias works better than 1w for 4h primary (faster response)
4. RSI 35-65 bands ensure trades happen while avoiding extremes
5. Position size 0.28 with discrete levels minimizes fee churn

Design:
- 1d KAMA(21) for macro trend direction (HTF filter)
- 4h KAMA(21) for primary trend + ADX(14) for strength
- 4h RSI(14) for pullback entries (35-65 range)
- ATR(14) 2.5x trailing stop for risk management
- Position size 0.28 (discrete: 0.0, ±0.28)
- Target: 25-50 trades/train (4 years), 6-12 trades/test (15 months)

Timeframe: 4h (as required by experiment)
HTF: 1d (daily trend bias)
Position Size: 0.28 (discrete levels to minimize fee churn)
Target: Sharpe > 0.618 (beat current best), DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_rsi_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market volatility - smooth in trends, responsive in ranges
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength (not direction)
    ADX > 25 = strong trend, ADX < 20 = weak/range
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        plus_diff = high[i] - high[i-1]
        minus_diff = low[i-1] - low[i]
        
        if plus_diff > minus_diff and plus_diff > 0:
            plus_dm[i] = plus_diff
        if minus_diff > plus_diff and minus_diff > 0:
            minus_dm[i] = minus_diff
    
    # Smooth with Wilder's method
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    # Initial sums
    plus_sum = np.sum(plus_dm[1:period+1])
    minus_sum = np.sum(minus_dm[1:period+1])
    tr_sum = np.sum(tr[1:period+1])
    
    for i in range(period, n):
        if i == period:
            plus_di[i] = 100.0 * plus_sum / tr_sum if tr_sum > 1e-10 else 0.0
            minus_di[i] = 100.0 * minus_sum / tr_sum if tr_sum > 1e-10 else 0.0
        else:
            plus_sum = plus_sum - plus_sum / period + plus_dm[i]
            minus_sum = minus_sum - minus_sum / period + minus_dm[i]
            tr_sum = tr_sum - tr_sum / period + tr[i]
            
            plus_di[i] = 100.0 * plus_sum / tr_sum if tr_sum > 1e-10 else 0.0
            minus_di[i] = 100.0 * minus_sum / tr_sum if tr_sum > 1e-10 else 0.0
        
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0.0
    
    # ADX = SMA of DX
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    return adx

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
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
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d KAMA for trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Appropriate size for 4h (25-50 trades/year target)
    
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
        if np.isnan(rsi[i]) or np.isnan(kama_4h[i]) or np.isnan(adx[i]):
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
        
        # === MACRO TREND (1d KAMA) - primary direction bias ===
        daily_bull = close[i] > kama_1d_aligned[i]
        daily_bear = close[i] < kama_1d_aligned[i]
        
        # === PRIMARY TREND (4h KAMA) - confirmation ===
        trend_bull = close[i] > kama_4h[i]
        trend_bear = close[i] < kama_4h[i]
        
        # === TREND STRENGTH (ADX) - filter weak trends ===
        # ADX > 20 = meaningful trend (not too strict to kill trades)
        trend_strong = adx[i] > 20.0
        
        # === RSI PULLBACK - LOOSE bands for MORE trades ===
        # Long: RSI pulled back but not oversold (35-55)
        rsi_pullback_long = 35.0 <= rsi[i] <= 55.0
        # Short: RSI rallied but not overbought (45-65)
        rsi_pullback_short = 45.0 <= rsi[i] <= 65.0
        
        # === DESIRED SIGNAL - BALANCED FOR TRADE COUNT ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + 4h bullish + ADX strong + RSI pullback
        if daily_bull and trend_bull and trend_strong and rsi_pullback_long:
            desired_signal = BASE_SIZE
        # LONG (looser): 1d bullish + 4h bullish + RSI pullback (no ADX filter)
        elif daily_bull and trend_bull and rsi_pullback_long:
            desired_signal = BASE_SIZE * 0.85
        # LONG (fallback): 1d bullish + 4h bullish + RSI not overbought
        elif daily_bull and trend_bull and rsi[i] < 60.0:
            desired_signal = BASE_SIZE * 0.70
        # LONG (weakest): 1d bullish + 4h above KAMA + ADX rising
        elif daily_bull and trend_bull and adx[i] > 18.0:
            desired_signal = BASE_SIZE * 0.55
        
        # SHORT: 1d bearish + 4h bearish + ADX strong + RSI pullback
        elif daily_bear and trend_bear and trend_strong and rsi_pullback_short:
            desired_signal = -BASE_SIZE
        # SHORT (looser): 1d bearish + 4h bearish + RSI pullback (no ADX filter)
        elif daily_bear and trend_bear and rsi_pullback_short:
            desired_signal = -BASE_SIZE * 0.85
        # SHORT (fallback): 1d bearish + 4h bearish + RSI not oversold
        elif daily_bear and trend_bear and rsi[i] > 40.0:
            desired_signal = -BASE_SIZE * 0.70
        # SHORT (weakest): 1d bearish + 4h below KAMA + ADX rising
        elif daily_bear and trend_bear and adx[i] > 18.0:
            desired_signal = -BASE_SIZE * 0.55
        
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
        if desired_signal >= BASE_SIZE * 0.9:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.7:
            final_signal = BASE_SIZE * 0.85
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.70
        elif desired_signal <= -BASE_SIZE * 0.9:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE * 0.85
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.70
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