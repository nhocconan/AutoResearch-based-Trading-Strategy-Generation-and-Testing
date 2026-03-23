#!/usr/bin/env python3
"""
Experiment #012: 12h Primary + 1d/1w HTF — KAMA Adaptive Trend + ADX + Choppiness Regime

Hypothesis: After 11 failed experiments, I'm testing KAMA (Kaufman Adaptive Moving Average)
which adapts to market noise better than HMA/EMA. Combined with ADX for trend strength
and Choppiness Index for regime detection, this should work better in bear/range markets.

Key differences from failed attempts:
1. KAMA instead of HMA — adapts efficiency ratio to volatility (better in chop)
2. ADX threshold = 20 (not 25+) — allows more trades while filtering noise
3. Choppiness as REGIME FILTER only (not primary signal like #001, #006)
4. RSI pullback entries (not CRSI which failed in #001, #006, #010)
5. Dual-mode: trend-follow when CHOP<50, mean-revert when CHOP>61.8
6. Position size 0.30 (conservative for 12h per Rule 4)

Why this might work:
- KAMA performed well in research for BTC/ETH through 2022 crash
- 12h TF targets 20-50 trades/year (fee-efficient, less churn)
- Regime-adaptive logic handles both trending and ranging markets
- Entry conditions LOOSE enough: RSI<45 or RSI>55 (not extreme 30/70)

Entry conditions (designed for 30-50 trades/year):
- Trend mode (CHOP<50): KAMA slope + ADX>20 + RSI pullback (45/55)
- Range mode (CHOP>61.8): RSI extremes (25/75) + BB mean reversion
- Either mode can trigger (not both required)

Stoploss: 3.0*ATR trailing, signal→0 when hit
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adx_chop_regime_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)."""
    close_s = pd.Series(close)
    n = len(close)
    
    # Change = absolute price change over period
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.abs(close[:period] - close[0])
    
    # Sum of individual changes (volatility)
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1))[1:])
    volatility[:period] = change[:period]
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    er[:period] = 0.0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Smooth using Wilder's method
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Avoid division by zero
    plus_di = plus_di / (atr + 1e-10) * 100
    minus_di = minus_di / (atr + 1e-10) * 100
    
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr_sum += max(high[j] - low[j], 
                         abs(high[j] - close[j-1]), 
                         abs(low[j] - close[j-1]))
        
        if highest_high - lowest_low > 0:
            chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    pct_b = (close - lower.values) / (upper.values - lower.values + 1e-10)
    
    return upper.values, lower.values, pct_b

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d KAMA for trend direction
    kama_1d = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 1w KAMA for macro regime
    kama_1w = calculate_kama(df_1w['close'].values, period=10)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate 12h indicators
    kama_12h = calculate_kama(close, period=10)
    adx_12h, plus_di_12h, minus_di_12h = calculate_adx(high, low, close, period=14)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    rsi_12h = calculate_rsi(close, period=14)
    bb_upper, bb_lower, bb_pct_b = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
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
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]):
            continue
        if np.isnan(kama_12h[i]) or np.isnan(adx_12h[i]) or np.isnan(chop_12h[i]):
            continue
        if np.isnan(atr_12h[i]) or np.isnan(rsi_12h[i]) or np.isnan(bb_pct_b[i]):
            continue
        if atr_12h[i] == 0:
            continue
        
        # === 1D TREND BIAS ===
        kama_1d_slope_bull = kama_1d_aligned[i] > kama_1d_aligned[i-2] if i >= 2 else False
        kama_1d_slope_bear = kama_1d_aligned[i] < kama_1d_aligned[i-2] if i >= 2 else False
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # === 1W MACRO REGIME ===
        price_above_kama_1w = close[i] > kama_1w_aligned[i] if not np.isnan(kama_1w_aligned[i]) else True
        price_below_kama_1w = close[i] < kama_1w_aligned[i] if not np.isnan(kama_1w_aligned[i]) else False
        
        # === 12H KAMA TREND ===
        kama_12h_slope_bull = kama_12h[i] > kama_12h[i-2] if i >= 2 else False
        kama_12h_slope_bear = kama_12h[i] < kama_12h[i-2] if i >= 2 else False
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_12h[i] > 20  # Lower threshold for more trades
        adx_weak = adx_12h[i] < 20
        
        # === CHOPPINESS REGIME ===
        chop_range = chop_12h[i] > 61.8  # Ranging market
        chop_trend = chop_12h[i] < 38.2  # Trending market
        chop_neutral = chop_12h[i] >= 38.2 and chop_12h[i] <= 61.8
        
        # === RSI CONDITIONS ===
        rsi_oversold = rsi_12h[i] < 35
        rsi_overbought = rsi_12h[i] > 65
        rsi_pullback_long = rsi_12h[i] < 45 and rsi_12h[i] > 30
        rsi_pullback_short = rsi_12h[i] > 55 and rsi_12h[i] < 70
        
        # === BB CONDITIONS ===
        bb_extreme_low = bb_pct_b[i] < 0.15
        bb_extreme_high = bb_pct_b[i] > 0.85
        
        # === DUAL REGIME LOGIC ===
        new_signal = 0.0
        
        # --- TREND MODE (CHOP < 50) ---
        if chop_12h[i] < 50 and adx_strong:
            # Long: KAMA bullish + pullback
            if kama_12h_slope_bull and price_above_kama_1d and rsi_pullback_long:
                new_signal = POSITION_SIZE
            
            # Short: KAMA bearish + rally
            if kama_12h_slope_bear and price_below_kama_1d and rsi_pullback_short:
                new_signal = -POSITION_SIZE
        
        # --- RANGE MODE (CHOP > 61.8) ---
        if chop_12h[i] > 61.8:
            # Long: RSI oversold + BB low
            if rsi_oversold and bb_extreme_low:
                # Only if not in strong 1w downtrend
                if price_above_kama_1w or not price_below_kama_1w:
                    new_signal = POSITION_SIZE
            
            # Short: RSI overbought + BB high
            if rsi_overbought and bb_extreme_high:
                # Only if not in strong 1w uptrend
                if price_below_kama_1w or not price_above_kama_1w:
                    new_signal = -POSITION_SIZE
        
        # --- NEUTRAL MODE (38.2 <= CHOP <= 61.8) ---
        if chop_neutral:
            # Simpler: follow 1d trend with RSI filter
            if kama_1d_slope_bull and rsi_12h[i] < 50:
                new_signal = POSITION_SIZE
            if kama_1d_slope_bear and rsi_12h[i] > 50:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (3.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * atr_12h[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * atr_12h[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if kama_1d_slope_bear and price_below_kama_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if kama_1d_slope_bull and price_above_kama_1d:
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