#!/usr/bin/env python3
"""
Experiment #174: 4h Primary + 12h/1d HTF — KAMA Trend + RSI Pullback + Regime Switch

Hypothesis: Previous strategies failed due to overly strict entry conditions (0 trades on BTC/ETH).
KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than HMA/EMA in choppy markets.
Combined with RSI pullback entries (not extremes) and Choppiness regime filter, this should
generate consistent trades across ALL symbols while maintaining positive Sharpe.

KEY IMPROVEMENTS:
1. KAMA (ER=10, fast=2, slow=30) - adapts to market volatility automatically
2. RSI(14) pullback entries: long at 35-45, short at 55-65 (NOT extremes = more trades)
3. Choppiness Index for regime: chop>55=range (mean revert), chop<45=trend (follow KAMA)
4. Donchian(20) breakout confirmation for momentum
5. Dual HTF bias: 12h KAMA for medium-term, 1d KAMA for macro
6. LOOSE entry: only 2 of 4 confluence factors needed (ensures trades on all symbols)
7. Position size: 0.30 full, 0.20 partial (discrete levels)
8. ATR trailing stop at 2.5x for risk management

TARGET: 30-50 trades/year, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_chop_donchian_12h1d_v1"
timeframe = "4h"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts to market volatility - smooth in trends, responsive in ranges.
    ER (Efficiency Ratio) determines smoothing constant.
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio: |change| / sum of absolute changes
    price_change = np.abs(close_s - close_s.shift(period)).values
    price_change[0:period] = np.nan
    
    sum_changes = np.zeros(len(close))
    for i in range(period, len(close)):
        sum_changes[i] = np.sum(np.abs(np.diff(close[max(0, i-period):i+1])))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = price_change / (sum_changes + 1e-10)
    
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0, 1)
    
    # Smoothing constant
    sc = (er * (2.0/(fast+1) - 2.0/(slow+1)) + 2.0/(slow+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bands)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Directional Movement
    plus_dm = np.maximum(high_s - high_s.shift(1), 0).values
    minus_dm = np.maximum(low_s.shift(1) - low, 0).values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Smooth with Wilder's method (EMA with span=period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    # DX and ADX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx = np.nan_to_num(adx, nan=0.0)
    
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_21 = calculate_kama(close, period=21, fast=2, slow=30)
    kama_50 = calculate_kama(close, period=50, fast=2, slow=30)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    # Calculate 12h KAMA for medium-term bias
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=21, fast=2, slow=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    # Calculate 1d KAMA for macro bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21, fast=2, slow=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Volume average (20-bar)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(kama_21[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(kama_12h_aligned[i]) or np.isnan(kama_1d_aligned[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        if np.isnan(adx[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === VOLUME FILTER (lenient) ===
        volume_ok = volume[i] > 0.5 * vol_avg[i]
        
        # === REGIME DETECTION ===
        chop_value = chop_14[i]
        is_trending = chop_value < 50.0  # Lenient threshold for more trades
        is_ranging = chop_value > 50.0
        
        # === HTF MACRO BIAS ===
        price_above_kama_12h = close[i] > kama_12h_aligned[i]
        price_below_kama_12h = close[i] < kama_12h_aligned[i]
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # === 4H TREND ===
        price_above_kama_21 = close[i] > kama_21[i]
        price_below_kama_21 = close[i] < kama_21[i]
        kama_21_slope_up = kama_21[i] > kama_21[i-1] if i > 0 else False
        kama_21_slope_down = kama_21[i] < kama_21[i-1] if i > 0 else False
        
        # === RSI PULLBACK (lenient thresholds for more trades) ===
        rsi_pullback_long = 35.0 <= rsi_14[i] <= 50.0  # Not extreme, more frequent
        rsi_pullback_short = 50.0 <= rsi_14[i] <= 65.0  # Not extreme, more frequent
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx[i] > 20.0  # Lenient threshold
        adx_weak = adx[i] < 25.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG entries - multiple confluence patterns (need 2+ of 4)
        long_signals = 0
        if rsi_pullback_long or rsi_oversold:
            long_signals += 1
        if price_above_kama_21 or kama_21_slope_up:
            long_signals += 1
        if price_above_kama_12h:
            long_signals += 1
        if donchian_breakout_long or volume_ok:
            long_signals += 1
        
        # LONG: Need 2+ signals (lenient for more trades)
        if long_signals >= 2:
            if is_trending and price_above_kama_1d:
                # Trend regime + macro bullish = full size
                new_signal = POSITION_SIZE_FULL
            elif is_ranging or price_above_kama_12h:
                # Range regime or 12h bias positive = half size
                new_signal = POSITION_SIZE_HALF
        
        # SHORT entries - multiple confluence patterns (need 2+ of 4)
        short_signals = 0
        if rsi_pullback_short or rsi_overbought:
            short_signals += 1
        if price_below_kama_21 or kama_21_slope_down:
            short_signals += 1
        if price_below_kama_12h:
            short_signals += 1
        if donchian_breakout_short or volume_ok:
            short_signals += 1
        
        # SHORT: Need 2+ signals (lenient for more trades)
        if short_signals >= 2:
            if is_trending and price_below_kama_1d:
                # Trend regime + macro bearish = full size
                new_signal = -POSITION_SIZE_FULL
            elif is_ranging or price_below_kama_12h:
                # Range regime or 12h bias negative = half size
                new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and trend still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price still above 4h KAMA
                if price_above_kama_21:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price still below 4h KAMA
                if price_below_kama_21:
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
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price crosses below 4h KAMA significantly
        if in_position and position_side > 0 and price_below_kama_21:
            new_signal = 0.0
        
        # Exit short if price crosses above 4h KAMA significantly
        if in_position and position_side < 0 and price_above_kama_21:
            new_signal = 0.0
        
        # Exit if macro bias flips strongly against position
        if in_position and position_side > 0 and price_below_kama_1d:
            new_signal = 0.0
        if in_position and position_side < 0 and price_above_kama_1d:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
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