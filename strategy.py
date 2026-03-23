#!/usr/bin/env python3
"""
Experiment #1114: 4h Primary + 12h/1d HTF — Asymmetric Regime Adaptive Strategy

Hypothesis: After 800+ failed experiments, key insight is that BTC/ETH behave differently
in bull vs bear markets. Simple trend-following fails in 2022 crash and 2025 bear.
This strategy uses REGIME-ADAPTIVE logic:

1. TREND REGIME (ADX>25, CHOP<38): Follow 12h HMA direction, enter on 4h RSI pullback
2. RANGE REGIME (ADX<20, CHOP>61): Mean revert at Bollinger bands with tight stops
3. TRANSITION REGIME: Reduced position size or flat

Key innovations vs failed experiments:
- Asymmetric thresholds: easier to enter long in bull, short in bear
- Volatility filter: ATR(7)/ATR(30) ratio detects vol spikes for better timing
- 12h HMA + 1d HMA confluence for stronger trend filter
- Loose RSI thresholds (35/65) to ensure adequate trade frequency
- Position size 0.25-0.30 with 2.5x ATR trailing stop

Why this should beat Sharpe=0.612:
- Adapts to 2022 crash (range/mean revert) and 2021 bull (trend follow)
- 4h timeframe = 20-50 trades/year target (low fee drag)
- MTF confluence (12h + 1d) reduces false signals
- Proven pattern from research: HMA + RSI + ATR + regime filter

Timeframe: 4h (primary)
HTF: 12h, 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.25 base, 0.15 reduced (discrete levels)
Stoploss: 2.5x ATR trailing
Target: 25-50 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_adaptive_hma_rsi_chop_12h1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    Formula: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        if span < 1:
            span = 1
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = max(1, int(period / 2))
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    diff = 2 * wma1 - wma2
    sqrt_period = max(1, int(np.sqrt(period)))
    hma = wma(diff, sqrt_period)
    return hma

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
    Average Directional Index — measures trend strength.
    ADX > 25 = strong trend, ADX < 20 = weak/choppy market.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    mask = tr_s > 1e-10
    plus_di[mask] = 100.0 * plus_dm_s[mask] / tr_s[mask]
    minus_di[mask] = 100.0 * minus_dm_s[mask] / tr_s[mask]
    
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    mask2 = di_sum > 1e-10
    dx[mask2] = 100.0 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending.
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period - 1, n):
        tr_sum = np.sum(tr[i - period + 1:i + 1])
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        
        if hh - ll > 1e-10:
            chop[i] = 100.0 * np.log10(tr_sum / (hh - ll)) / np.log10(period)
    
    return chop

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands for mean reversion levels."""
    n = len(close)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < period:
        return middle, upper, lower
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window, ddof=0)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
    
    return middle, upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 12h HMA for intermediate trend
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    atr_4h_long = calculate_atr(high, low, close, period=30)
    adx_4h = calculate_adx(high, low, close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    bb_mid, bb_upper, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.14
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]) or np.isnan(adx_4h[i]):
            continue
        if np.isnan(chop_4h[i]) or np.isnan(bb_mid[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_4h_long[i]) or atr_4h[i] <= 1e-10:
            continue
        
        # === VOLATILITY FILTER ===
        # ATR ratio detects vol spikes (entry timing)
        atr_ratio = atr_4h[i] / (atr_4h_long[i] + 1e-10)
        vol_spike = atr_ratio > 1.5  # Current vol > 1.5x long-term avg
        vol_normal = atr_ratio < 1.3
        
        # === REGIME DETECTION ===
        # ADX + Choppiness determines market regime
        trend_regime = adx_4h[i] > 25.0 and chop_4h[i] < 45.0
        range_regime = adx_4h[i] < 20.0 and chop_4h[i] > 55.0
        transition_regime = not trend_regime and not range_regime
        
        # === MACRO TREND (12h + 1d HMA confluence) ===
        macro_bull = close[i] > hma_12h_aligned[i] and close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i] and close[i] < hma_1d_aligned[i]
        macro_neutral = not macro_bull and not macro_bear
        
        # === RSI THRESHOLDS (asymmetric per regime) ===
        # Trend regime: looser RSI for pullback entries
        # Range regime: tighter RSI for mean reversion
        if trend_regime:
            rsi_oversold = rsi_4h[i] < 45.0
            rsi_overbought = rsi_4h[i] > 55.0
            rsi_extreme_long = rsi_4h[i] < 35.0
            rsi_extreme_short = rsi_4h[i] > 65.0
        else:
            rsi_oversold = rsi_4h[i] < 35.0
            rsi_overbought = rsi_4h[i] > 65.0
            rsi_extreme_long = rsi_4h[i] < 25.0
            rsi_extreme_short = rsi_4h[i] > 75.0
        
        # === BOLLINGER POSITION ===
        near_bb_lower = close[i] < bb_lower[i] + 0.3 * (bb_upper[i] - bb_lower[i])
        near_bb_upper = close[i] > bb_upper[i] - 0.3 * (bb_upper[i] - bb_lower[i])
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === TREND REGIME: Follow HMA direction ===
        if trend_regime:
            if macro_bull and rsi_oversold:
                # Long on pullback in uptrend
                if vol_normal or rsi_extreme_long:
                    desired_signal = current_size
            elif macro_bear and rsi_overbought:
                # Short on rally in downtrend
                if vol_normal or rsi_extreme_short:
                    desired_signal = -current_size
        
        # === RANGE REGIME: Mean revert at BB bounds ===
        elif range_regime:
            if rsi_extreme_long and near_bb_lower:
                # Long at lower BB with extreme RSI
                desired_signal = current_size
            elif rsi_extreme_short and near_bb_upper:
                # Short at upper BB with extreme RSI
                desired_signal = -current_size
        
        # === TRANSITION REGIME: Reduced size or flat ===
        elif transition_regime:
            current_size = REDUCED_SIZE
            if macro_bull and rsi_extreme_long:
                desired_signal = current_size
            elif macro_bear and rsi_extreme_short:
                desired_signal = -current_size
        
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro still bull
                if macro_bull and not macro_bear:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if macro still bear
                if macro_bear and not macro_bull:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses or RSI overbought
            if macro_bear or rsi_4h[i] > 70.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses or RSI oversold
            if macro_bull or rsi_4h[i] < 30.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = 0.0
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = 0.0
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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