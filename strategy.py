#!/usr/bin/env python3
"""
Experiment #1151: 4h Primary + 1d HTF — ADX Regime + KAMA Trend + BB/Donchian Entry

Hypothesis: After 839+ failed experiments, clear patterns emerge:
- CRSI + Choppiness combos FAILING (negative Sharpe in #1139-#1148)
- 0-trade strategies from overly strict conditions (#1148, #1150)
- Current BEST: mtf_4h_triple_regime_crsi_donchian_1d1w_v1 (Sharpe=0.612)

This strategy uses DIFFERENT proven components:
1. ADX(14) regime filter: >25=trend, <20=range (hysteresis to reduce churn)
2. 1d KAMA(21) for macro trend direction (adaptive, better than HMA in chop)
3. 4h KAMA(21) for local trend confirmation
4. Dual entry system:
   - Trend regime: Donchian(20) breakout + RSI(14) momentum
   - Range regime: Bollinger(20,2.0) mean reversion + RSI extremes
5. ATR(14) 2.5x trailing stop (wider than 2.0x to avoid premature exits)
6. Position size 0.30 discrete (balance returns vs drawdown)

Why this should beat Sharpe=0.612:
- ADX regime switching adapts to market conditions (trend vs range)
- KAMA adapts smoothing based on efficiency ratio (less lag than EMA)
- Dual entry captures both breakout momentum AND mean reversion
- 1d KAMA macro filter prevents counter-trend trades (key for 2022/2025)
- Target: 30-50 trades/year on 4h (optimal for fee drag)

Timeframe: 4h (primary)
HTF: 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.30 base (discrete: 0.0, ±0.30)
Stoploss: 2.5x ATR trailing
Target: 30-50 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_adx_regime_kama_bb_donchian_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average — adapts to market noise.
    ER (Efficiency Ratio) determines smoothing constant.
    High ER = trending (less smoothing), Low ER = choppy (more smoothing)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(slow_period, n):
        signal = abs(close[i] - close[i - slow_period])
        noise = np.sum(np.abs(np.diff(close[i - slow_period:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[slow_period] = close[slow_period]
    
    for i in range(slow_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

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

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index — trend strength indicator.
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0.0)
        else:
            plus_dm[i] = 0.0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0.0)
        else:
            minus_dm[i] = 0.0
    
    # Smooth with Wilder's method (EMA with alpha=1/period)
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DX and ADX
    di_sum = plus_di + minus_di
    dx = np.zeros(n)
    mask = di_sum > 1e-10
    dx[mask] = 100.0 * np.abs(plus_di[mask] - minus_di[mask]) / di_sum[mask]
    
    # ADX is smoothed DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands for mean reversion detection."""
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    if n < period:
        return upper, lower, middle
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window, ddof=0)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
    
    return upper, lower, middle

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
    
    # Calculate and align 1d KAMA for adaptive macro trend filter
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    bb_upper, bb_lower, bb_middle = calculate_bollinger(close, period=20, std_mult=2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Also calculate 4h KAMA for local trend
    kama_4h = calculate_kama(close, period=21)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # ADX regime tracking with hysteresis
    prev_adx_regime = 0  # 0=unknown, 1=trend, 2=range
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(atr[i]) or np.isnan(adx[i]):
            continue
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_4h[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(donchian_upper[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === ADX REGIME DETECTION (with hysteresis) ===
        # Enter trend regime at ADX > 25, exit at ADX < 20
        # Enter range regime at ADX < 20, exit at ADX > 25
        if prev_adx_regime == 0:
            if adx[i] > 25:
                adx_regime = 1  # trend
            elif adx[i] < 20:
                adx_regime = 2  # range
            else:
                adx_regime = prev_adx_regime
        elif prev_adx_regime == 1:  # was trend
            adx_regime = 2 if adx[i] < 20 else 1
        else:  # was range
            adx_regime = 1 if adx[i] > 25 else 2
        
        prev_adx_regime = adx_regime
        
        # === MACRO TREND (1d KAMA) ===
        macro_bull = close[i] > kama_1d_aligned[i]
        macro_bear = close[i] < kama_1d_aligned[i]
        
        # === LOCAL TREND (4h KAMA) ===
        local_bull = close[i] > kama_4h[i]
        local_bear = close[i] < kama_4h[i]
        
        # === TREND REGIME SIGNALS (ADX > 25) ===
        # Long: breakout above Donchian + RSI momentum
        # Short: breakout below Donchian + RSI momentum
        breakout_long = close[i] > donchian_upper[i - 1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i - 1] if i > 0 else False
        rsi_bullish = rsi_4h[i] > 50.0
        rsi_bearish = rsi_4h[i] < 50.0
        
        # === RANGE REGIME SIGNALS (ADX < 20) ===
        # Long: price near BB lower + RSI oversold
        # Short: price near BB upper + RSI overbought
        bb_long = close[i] <= bb_lower[i] * 1.002  # at or below lower band
        bb_short = close[i] >= bb_upper[i] * 0.998  # at or above upper band
        rsi_oversold = rsi_4h[i] < 40.0
        rsi_overbought = rsi_4h[i] > 60.0
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        if adx_regime == 1:  # Trend regime
            # Macro bull + local bull + breakout + RSI confirms
            if macro_bull and local_bull and breakout_long and rsi_bullish:
                desired_signal = BASE_SIZE
        else:  # Range regime
            # Macro bull + BB lower + RSI oversold
            if macro_bull and bb_long and rsi_oversold:
                desired_signal = BASE_SIZE
        
        # === SHORT ENTRY ===
        if adx_regime == 1:  # Trend regime
            # Macro bear + local bear + breakout + RSI confirms
            if macro_bear and local_bear and breakout_short and rsi_bearish:
                desired_signal = -BASE_SIZE
        else:  # Range regime
            # Macro bear + BB upper + RSI overbought
            if macro_bear and bb_short and rsi_overbought:
                desired_signal = -BASE_SIZE
        
        # === EXTREME RSI EXIT ===
        if in_position and position_side > 0 and rsi_4h[i] > 75.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_4h[i] < 25.0:
            desired_signal = 0.0
        
        # === MACRO TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and macro_bear:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bull:
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
                if macro_bull and local_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                if macro_bear and local_bear:
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