#!/usr/bin/env python3
"""
Experiment #051: 6h Primary + 1w/1d HTF — KAMA Adaptive Trend + RSI Pullback + ADX Filter

Hypothesis: 6h is a unique timeframe between 4h and 12h that needs adaptive filtering.
Previous 6h strategies failed because:
- CRSI approaches failed (#040, #044) - too mean-reversion focused
- Weekly pivot strategies failed (#043) - too rigid at specific levels
- Vol spike strategies failed (#047) - too rare triggers

NEW APPROACH:
- KAMA (Kaufman Adaptive MA) responds to volatility regimes automatically
- 1w HMA for MAJOR trend bias (weekly trends more persistent than daily)
- 1d HMA for secondary confirmation (dual HTF filter)
- RSI(7) for pullback entries (faster than RSI14, catches 6h dips)
- ADX(14) > 20 filter to avoid choppy whipsaws (proven in literature)
- ATR-based stoploss at 2.5x (tighter for 6h swings)

Key design choices:
- Timeframe: 6h (30-60 trades/year target)
- HTF: 1w HMA (major bias) + 1d HMA (confirmation)
- Entry: KAMA cross + RSI pullback + ADX trend strength + HTF alignment
- Position size: 0.30 (30% of capital, standard for 6h)
- Stoploss: 2.5x ATR trailing
- LOOSE enough filters to ensure >=30 trades on train, >=3 on test

Why this might work when others failed:
- KAMA adapts to volatility (unlike fixed EMA/HMA)
- Dual HTF (1w + 1d) provides stronger trend confirmation than single HTF
- RSI(7) catches pullbacks faster than RSI(14)
- ADX filter avoids the choppy periods that destroyed previous 6h strategies
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_rsi_adx_dual_htf_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (volatility vs trend)
    ER = Efficiency Ratio = |Net Change| / Sum of Absolute Changes
    SC = Smoothing Constant = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        net_change = abs(close[i] - close[i - er_period])
        sum_changes = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0.0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength (not direction)
    ADX > 25 = strong trend, ADX < 20 = weak/choppy
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth with Wilder's method (EMA with alpha = 1/period)
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(period, n):
        if atr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / atr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / atr_smooth[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d HMA for secondary confirmation
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 6h pullbacks
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    
    # Also calculate 6h HMA for cross confirmation
    hma_6h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (standard for 6h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_6h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (Dual: 1w + 1d) ===
        htf_bull = (close[i] > hma_1w_aligned[i]) and (close[i] > hma_1d_aligned[i])
        htf_bear = (close[i] < hma_1w_aligned[i]) and (close[i] < hma_1d_aligned[i])
        htf_neutral = not htf_bull and not htf_bear
        
        # === TREND STRENGTH (ADX) ===
        # ADX > 20 = enough trend to trade, ADX > 25 = strong trend
        adx_strong = adx[i] > 25.0
        adx_ok = adx[i] > 20.0
        
        # === KAMA ADAPTIVE TREND ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === 6h HMA CROSS ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === RSI PULLBACK (faster RSI for 6h) ===
        # Long: RSI dipped to 35-45 then recovering
        # Short: RSI rose to 55-65 then falling
        rsi_oversold_pullback = 30.0 < rsi[i] < 50.0
        rsi_overbought_pullback = 50.0 < rsi[i] < 70.0
        rsi_extreme_oversold = rsi[i] < 35.0
        rsi_extreme_overbought = rsi[i] > 65.0
        
        # === DESIRED SIGNAL (KAMA + RSI Pullback + ADX + HTF) ===
        desired_signal = 0.0
        
        # LONG entries
        if kama_bull and hma_bull and adx_ok:
            # Primary: RSI pullback in uptrend with HTF confirmation
            if rsi_oversold_pullback and htf_bull:
                desired_signal = SIZE
            # Secondary: RSI extreme oversold (stronger signal, can override neutral HTF)
            elif rsi_extreme_oversold and (htf_bull or htf_neutral):
                desired_signal = SIZE * 0.8
            # Tertiary: KAMA cross up with strong ADX (breakout style)
            elif close[i] > kama[i] and close[i-1] <= kama[i-1] and adx_strong and htf_bull:
                desired_signal = SIZE * 0.7
        
        # SHORT entries
        elif kama_bear and hma_bear and adx_ok:
            # Primary: RSI pullback in downtrend with HTF confirmation
            if rsi_overbought_pullback and htf_bear:
                desired_signal = -SIZE
            # Secondary: RSI extreme overbought (stronger signal, can override neutral HTF)
            elif rsi_extreme_overbought and (htf_bear or htf_neutral):
                desired_signal = -SIZE * 0.8
            # Tertiary: KAMA cross down with strong ADX (breakout style)
            elif close[i] < kama[i] and close[i-1] >= kama[i-1] and adx_strong and htf_bear:
                desired_signal = -SIZE * 0.7
        
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
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.6:
            final_signal = SIZE * 0.7
        elif desired_signal <= -SIZE * 0.6:
            final_signal = -SIZE * 0.7
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