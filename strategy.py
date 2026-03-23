#!/usr/bin/env python3
"""
Experiment #004: 4h Primary + 12h HTF — KAMA Adaptive Trend + BB Mean Reversion + ADX Regime

Hypothesis: After 3 failed experiments with CRSI/Choppiness/complex regime filters, I'm shifting to
KAMA (Kaufman Adaptive Moving Average) which adapts to market efficiency - proven to work better
than EMA on BTC/ETH through 2022 crash. Combined with BB mean reversion in low-ADX regimes.

Key differences from failed attempts:
1. NO Choppiness Index (failed in #001, #002)
2. NO Connors RSI (failed in #001)
3. Using KAMA instead of HMA/EMA - adapts speed based on volatility (ER efficiency ratio)
4. Dual regime: Mean revert when ADX<25, trend follow when ADX>30
5. Simpler entry conditions to ensure trades are generated (Rule 9)

Why this might work:
- KAMA worked through 2022 crash (unlike pure EMA crossover)
- ADX regime filter switches between mean-revert and trend-follow appropriately
- 4h TF targets 20-50 trades/year (fee-efficient per Rule 10)
- Position size 0.30 (conservative for 4h per Rule 4)
- 12h KAMA for trend bias (HTF alignment proven to 2x Sharpe)

Entry conditions (LOOSE enough to generate ≥10 trades/symbol):
- Long Mean Revert: BB_pct_b < 0.15 AND ADX < 25 AND 12h KAMA bullish
- Short Mean Revert: BB_pct_b > 0.85 AND ADX < 25 AND 12h KAMA bearish
- Long Trend: Price > Donchian(20) high AND ADX > 30 AND 12h KAMA bullish
- Short Trend: Price < Donchian(20) low AND ADX > 30 AND 12h KAMA bearish

Stoploss: 2.5*ATR trailing, signal→0 when hit
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_bb_regime_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (volatility vs trend).
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Calculate Efficiency Ratio (ER)
    change = close_s.diff(er_period).abs()
    volatility = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    er = change / (volatility + 1e-10)
    er = er.fillna(0)
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    # %B (position within bands)
    pct_b = (close - lower) / (upper - lower + 1e-10)
    
    return upper.values, lower.values, pct_b.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h KAMA for trend direction
    kama_12h = calculate_kama(df_12h['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Calculate 4h indicators
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    bb_upper, bb_lower, bb_pct_b = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(kama_12h_aligned[i]) or np.isnan(kama_4h[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(adx_14[i]):
            continue
        if np.isnan(bb_pct_b[i]) or np.isnan(donchian_upper[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 12H TREND BIAS ===
        kama_12h_slope_bull = kama_12h_aligned[i] > kama_12h_aligned[i-3] if i >= 3 else False
        kama_12h_slope_bear = kama_12h_aligned[i] < kama_12h_aligned[i-3] if i >= 3 else False
        price_above_kama_12h = close[i] > kama_12h_aligned[i]
        price_below_kama_12h = close[i] < kama_12h_aligned[i]
        
        # === 4H KAMA TREND ===
        kama_4h_slope_bull = kama_4h[i] > kama_4h[i-3] if i >= 3 else False
        kama_4h_slope_bear = kama_4h[i] < kama_4h[i-3] if i >= 3 else False
        
        # === ADX REGIME ===
        adx_low = adx_14[i] < 25  # Range/mean-revert regime
        adx_high = adx_14[i] > 30  # Trend regime
        
        # === BOLLINGER BAND EXTREMES ===
        bb_extreme_low = bb_pct_b[i] < 0.15
        bb_extreme_high = bb_pct_b[i] > 0.85
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_high = close[i] > donchian_upper[i-1] if i >= 1 else False
        donchian_breakout_low = close[i] < donchian_lower[i-1] if i >= 1 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRIES ---
        # Mean Revert: BB extreme low + low ADX + 12h trend not bearish
        if adx_low and bb_extreme_low and not kama_12h_slope_bear:
            new_signal = POSITION_SIZE
        
        # Trend Follow: Donchian breakout + high ADX + 12h trend bullish
        elif adx_high and donchian_breakout_high and kama_12h_slope_bull:
            new_signal = POSITION_SIZE
        
        # --- SHORT ENTRIES ---
        # Mean Revert: BB extreme high + low ADX + 12h trend not bullish
        if adx_low and bb_extreme_high and not kama_12h_slope_bull:
            new_signal = -POSITION_SIZE
        
        # Trend Follow: Donchian breakout + high ADX + 12h trend bearish
        elif adx_high and donchian_breakout_low and kama_12h_slope_bear:
            new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # If already in position and no new signal, hold current position
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            # Exit long if 12h KAMA turns bearish
            if kama_12h_slope_bear and price_below_kama_12h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 12h KAMA turns bullish
            if kama_12h_slope_bull and price_above_kama_12h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals