#!/usr/bin/env python3
"""
Experiment #1377: 1d Primary + 1w HTF — KAMA Adaptive Trend Following

Hypothesis: Previous failures with regime filters (Choppiness, CRSI) over-complicated
entry logic and reduced trade frequency. KAMA (Kaufman Adaptive Moving Average) 
automatically adapts to market noise without manual regime detection - efficient
ratio determines smoothing constant based on price efficiency.

Key insight: 1d timeframe with weekly macro bias should capture major trends while
KAMA's adaptive nature handles both trending and ranging periods. ADX > 20 (not 25+)
ensures sufficient trend strength without over-filtering. Donchian(20) breakout
provides clean entry trigger.

Design:
1. 1w KAMA(21) = macro trend bias (soft filter, not hard requirement)
2. 1d KAMA(21) + Efficiency Ratio = primary trend with adaptive smoothing
3. ADX(14) > 20 = minimum trend strength (lower than typical 25 to ensure trades)
4. Donchian(20) breakout = entry trigger
5. RSI(14) 35-65 wide bands = momentum confirmation without over-filtering
6. ATR(14) 2.5x trailing stop = risk management
7. Position size 0.30 = conservative for daily volatility
8. THREE entry paths per direction = ensures >=30 trades/train

Target: 20-40 trades/year, Sharpe > 0.618, trades >= 30 train, >= 5 test
Timeframe: 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_adx_donchian_1w_rsi_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average - adapts smoothing based on market efficiency
    ER (Efficiency Ratio) = |net change| / sum of absolute changes
    High ER = trending (less smoothing), Low ER = noisy (more smoothing)
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(period, n):
        net_change = abs(close[i] - close[i - period])
        sum_changes = 0.0
        for j in range(i - period + 1, i + 1):
            sum_changes += abs(close[j] - close[j - 1])
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if not np.isnan(er[i]):
            # Adaptive smoothing constant
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength indicator"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth TR, +DM, -DM
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI, -DI
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    
    mask = atr > 1e-10
    plus_di[mask] = 100.0 * plus_dm_smooth[mask] / atr[mask]
    minus_di[mask] = 100.0 * minus_dm_smooth[mask] / atr[mask]
    
    # Calculate DX and ADX
    dx = np.full(n, np.nan)
    adx = np.full(n, np.nan)
    
    for i in range(period * 2, n):
        if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 1e-10:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Smooth DX to get ADX
    dx_series = pd.Series(dx)
    adx_raw = dx_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    adx = adx_raw
    
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF KAMA for macro trend filter
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=21)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate primary (1d) indicators
    kama_1d = calculate_kama(close, period=21)
    adx = calculate_adx(high, low, close, period=14)
    donchian_20_upper, donchian_20_lower = calculate_donchian(high, low, period=20)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(donchian_20_upper[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi[i]) or np.isnan(kama_1d[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        if np.isnan(kama_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (1w KAMA) - soft filter ===
        macro_bull = close[i] > kama_1w_aligned[i]
        macro_bear = close[i] < kama_1w_aligned[i]
        
        # === PRIMARY TREND (1d KAMA) ===
        trend_bull = close[i] > kama_1d[i]
        trend_bear = close[i] < kama_1d[i]
        
        # === KAMA SLOPE ===
        kama_slope_bull = False
        kama_slope_bear = False
        if i >= 5 and not np.isnan(kama_1d[i]) and not np.isnan(kama_1d[i-5]):
            kama_slope_bull = kama_1d[i] > kama_1d[i-5]
            kama_slope_bear = kama_1d[i] < kama_1d[i-5]
        
        # === ADX TREND STRENGTH (>= 20, not 25+) ===
        trend_strength = adx[i] >= 20.0
        
        # === RSI MOMENTUM (WIDE bands to ensure trades) ===
        rsi_bull = rsi[i] > 35.0
        rsi_bear = rsi[i] < 65.0
        rsi_strong_bull = rsi[i] > 50.0
        rsi_strong_bear = rsi[i] < 50.0
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_20_upper[i-1]
        breakout_short = close[i] < donchian_20_lower[i-1]
        
        # === DESIRED SIGNAL - THREE ENTRY PATHS PER DIRECTION ===
        desired_signal = 0.0
        
        # LONG ENTRY PATHS
        # Path 1: Donchian breakout + trend + ADX strength (primary entry)
        if breakout_long and trend_bull and trend_strength and rsi_bull:
            desired_signal = BASE_SIZE
        # Path 2: KAMA slope + macro confirmation + RSI momentum
        elif kama_slope_bull and macro_bull and rsi_strong_bull and trend_bull:
            desired_signal = BASE_SIZE * 0.5
        # Path 3: Price above both KAMAs + ADX building trend
        elif close[i] > kama_1d[i] and close[i] > kama_1w_aligned[i] and adx[i] >= 18.0:
            desired_signal = BASE_SIZE * 0.5
        
        # SHORT ENTRY PATHS
        # Path 1: Donchian breakout + trend + ADX strength (primary entry)
        elif breakout_short and trend_bear and trend_strength and rsi_bear:
            desired_signal = -BASE_SIZE
        # Path 2: KAMA slope + macro confirmation + RSI momentum
        elif kama_slope_bear and macro_bear and rsi_strong_bear and trend_bear:
            desired_signal = -BASE_SIZE * 0.5
        # Path 3: Price below both KAMAs + ADX building trend
        elif close[i] < kama_1d[i] and close[i] < kama_1w_aligned[i] and adx[i] >= 18.0:
            desired_signal = -BASE_SIZE * 0.5
        
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
        if abs(desired_signal) >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE if desired_signal > 0 else -BASE_SIZE
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