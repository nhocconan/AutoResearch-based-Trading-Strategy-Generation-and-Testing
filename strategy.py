#!/usr/bin/env python3
"""
Experiment #1156: 12h Primary + 1d HTF — KAMA Adaptive Trend + Donchian Breakout + RSI Filter

Hypothesis: After 842+ failed experiments, clear patterns emerge:
- CRSI + Choppiness regime switching FAILS consistently (negative Sharpe #1144-#1148)
- Simple trend + breakout WORKS (#1149 Sharpe=0.050, #1153 Sharpe=0.299, #1154 Sharpe=0.150)
- KAMA (Kaufman Adaptive) adapts to market regime automatically — no manual regime detection needed
- 12h timeframe targets 20-50 trades/year (optimal fee drag vs signal quality)

This strategy uses PROVEN components with NEW combination:
1. 1d HMA(21) for macro trend direction (smoother than 12h, less whipsaw)
2. 12h KAMA(10,2,30) for adaptive trend following (works in BOTH ranging AND trending)
3. 12h Donchian(20) breakout for entry timing (catches momentum bursts)
4. 12h RSI(14) moderate filter (30-70 range, NOT extreme — avoids 0 trades)
5. 12h ADX(14) > 15 for trend confirmation (lower threshold = more trades)
6. 12h ATR(14) 2.5x trailing stop (protects gains, wider than 2.0x for 12h TF)
7. Position size 0.30 discrete (balance returns vs drawdown)

Why this should beat Sharpe=0.612:
- KAMA adapts efficiency ratio automatically — no manual regime detection (failed in #1146-#1148)
- 1d HMA macro filter is smoother than 12h (less false reversals)
- RSI 30-70 filter is MODERATE (not extreme 20/80) — ensures we get trades
- ADX > 15 (not > 25) — more trade opportunities while filtering chop
- Target: 25-45 trades/year on 12h (optimal for fee drag)

Timeframe: 12h (primary)
HTF: 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.30 base (discrete: 0.0, ±0.30)
Stoploss: 2.5x ATR trailing (wider for 12h TF)
Target: 25-45 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_donchian_rsi_adx_1d_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average — adapts to market noise.
    KAMA adjusts smoothing constant based on Efficiency Ratio (ER).
    ER = |net change| / sum of absolute changes (0=noise, 1=trend)
    Fast SC = 2/(fast+1), Slow SC = 2/(slow+1)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(er_period - 1, n):
        net_change = abs(close[i] - close[i - er_period + 1])
        sum_changes = np.sum(np.abs(np.diff(close[i - er_period + 1:i + 1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period - 1] = close[er_period - 1]
    
    # Calculate KAMA
    for i in range(er_period, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index — measures trend strength.
    ADX > 20 = trend, ADX < 15 = choppy/ranging
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
        else:
            minus_dm[i] = 0
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    mask = tr_smooth > 1e-10
    di_plus[mask] = 100.0 * plus_dm_smooth[mask] / tr_smooth[mask]
    di_minus[mask] = 100.0 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    dx = np.zeros(n)
    for i in range(period * 2 - 1, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(di_plus[i] - di_minus[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    
    return rsi

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel — breakout detection.
    Upper = highest high over period
    Lower = lowest low over period
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    adx_12h = calculate_adx(high, low, close, period=14)
    rsi_12h = calculate_rsi(close, period=14)
    kama_12h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
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
        if np.isnan(atr[i]) or np.isnan(adx_12h[i]) or np.isnan(rsi_12h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(kama_12h[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === ADAPTIVE TREND (12h KAMA) ===
        # KAMA adapts to market — price above KAMA = bull, below = bear
        kama_bull = close[i] > kama_12h[i]
        kama_bear = close[i] < kama_12h[i]
        
        # === KAMA DIRECTION ===
        # KAMA sloping up = bull, sloping down = bear
        kama_rising = False
        kama_falling = False
        if i >= 5 and not np.isnan(kama_12h[i-5]):
            kama_rising = kama_12h[i] > kama_12h[i-5]
            kama_falling = kama_12h[i] < kama_12h[i-5]
        
        # === BREAKOUT SIGNAL (Donchian) ===
        breakout_long = close[i] > donchian_upper[i - 1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i - 1] if i > 0 else False
        
        # === RSI FILTER (MODERATE — ensures trades) ===
        # Long: RSI > 45 (momentum building, not oversold trap)
        # Short: RSI < 55 (momentum weakening, not overbought trap)
        rsi_ok_long = rsi_12h[i] > 45.0
        rsi_ok_short = rsi_12h[i] < 55.0
        
        # === TREND STRENGTH (ADX) ===
        # ADX > 15 confirms trend has some momentum (lower threshold = more trades)
        trend_strong = adx_12h[i] > 15.0
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Macro bull + KAMA bull + KAMA rising + breakout + RSI ok + ADX confirms
        if macro_bull and kama_bull and kama_rising and breakout_long and rsi_ok_long and trend_strong:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY ===
        # Macro bear + KAMA bear + KAMA falling + breakout + RSI ok + ADX confirms
        elif macro_bear and kama_bear and kama_falling and breakout_short and rsi_ok_short and trend_strong:
            desired_signal = -BASE_SIZE
        
        # === MACRO TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and macro_bear:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bull:
            desired_signal = 0.0
        
        # === KAMA REVERSAL EXIT ===
        if in_position and position_side > 0 and kama_bear:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and kama_bull:
            desired_signal = 0.0
        
        # === ADX WEAKNESS EXIT ===
        # If ADX drops below 12, trend is weakening — exit
        if in_position and adx_12h[i] < 12.0:
            desired_signal = 0.0
        
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                if macro_bull and kama_bull and adx_12h[i] >= 12.0:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                if macro_bear and kama_bear and adx_12h[i] >= 12.0:
                    desired_signal = -BASE_SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals