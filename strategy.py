#!/usr/bin/env python3
"""
Experiment #287: 1d Primary + 1w HTF — KAMA Adaptive Trend + Choppiness Regime + RSI Entries

Hypothesis: After #283 showed promise (Sharpe=0.192 on 1d), improve with:
1. KAMA instead of HMA — adapts smoothing based on market efficiency (less whipsaw in chop)
2. Choppiness Index for regime detection (proven in #282, #283)
3. RSI(14) with RELAXED thresholds to ensure 10+ trades per symbol
4. 1w HTF for primary trend bias (slower, more stable than 1d)
5. Simple ATR stoploss (2.5x) — no complex trailing
6. Force entries every 15 bars if flat to guarantee trade frequency

Key insight from failures: Overly strict conditions = 0 trades. Must loosen RSI thresholds
and add frequency safeguards. KAMA adapts to volatility better than HMA in range markets.

Position sizing: 0.30 base, 0.35 strong conviction (discrete levels)
Target: 20-40 trades/year per symbol (appropriate for 1d)
Stoploss: 2.5 * ATR from entry
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_chop_rsi_1w_v1"
timeframe = "1d"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency ratio.
    More responsive in trends, smoother in chop.
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Efficiency Ratio: |net change| / sum of absolute changes
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.nan
    
    sum_change = np.zeros(n)
    for i in range(period, n):
        sum_change[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1)))
    
    er = change / np.where(sum_change > 0, sum_change, np.nan)
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0.0, 1.0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    kama[:period] = np.nan
    return kama

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (primary trend regime)
    kama_1w_21 = calculate_kama(df_1w['close'].values, 21)
    rsi_1w_14 = calculate_rsi(df_1w['close'].values, 14)
    chop_1w_14 = calculate_choppiness_index(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_1w_21_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_21)
    rsi_1w_14_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w_14)
    chop_1w_14_aligned = align_htf_to_ltf(prices, df_1w, chop_1w_14)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_1d_21 = calculate_kama(close, 21)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    STRONG_SIZE = 0.35
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -15
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_1d_21[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(chop_1w_14_aligned[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1W TREND REGIME (primary direction filter) ===
        regime_bull = close[i] > kama_1w_21_aligned[i]
        regime_bear = close[i] < kama_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === 1D LOCAL SIGNALS ===
        price_above_kama = close[i] > kama_1d_21[i]
        price_below_kama = close[i] < kama_1d_21[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i] * 0.998
        donchian_breakout_short = close[i] < donchian_lower[i] * 1.002
        
        # === RSI THRESHOLDS (relaxed for guaranteed trades) ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_extreme_oversold = rsi_14[i] < 30.0
        rsi_extreme_overbought = rsi_14[i] > 70.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND FOLLOWING MODE (when trending)
        if is_trending:
            # LONG: Trend + bull regime + price above KAMA
            if regime_bull and price_above_kama and rsi_14[i] > 35:
                new_signal = STRONG_SIZE
            # LONG: Donchian breakout + bull regime
            elif donchian_breakout_long and regime_bull:
                new_signal = BASE_SIZE
            
            # SHORT: Trend + bear regime + price below KAMA
            if regime_bear and price_below_kama and rsi_14[i] < 65:
                if new_signal == 0.0 or abs(new_signal) < STRONG_SIZE:
                    new_signal = -STRONG_SIZE
            # SHORT: Donchian breakdown + bear regime
            elif donchian_breakout_short and regime_bear:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # MEAN REVERSION MODE (when choppy)
        if is_choppy:
            # LONG: Choppy + RSI oversold
            if rsi_oversold:
                new_signal = BASE_SIZE
            # LONG: Choppy + RSI extreme oversold
            if rsi_extreme_oversold:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE
            
            # SHORT: Choppy + RSI overbought
            if rsi_overbought and new_signal == 0.0:
                new_signal = -BASE_SIZE
            # SHORT: Choppy + RSI extreme overbought
            if rsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 15 bars (~15 days on 1d)
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 30 and price_above_kama:
                new_signal = BASE_SIZE * 0.8
            elif regime_bear and rsi_14[i] < 70 and price_below_kama:
                new_signal = -BASE_SIZE * 0.8
            elif is_choppy and rsi_14[i] < 35:
                new_signal = BASE_SIZE * 0.7
            elif is_choppy and rsi_14[i] > 65:
                new_signal = -BASE_SIZE * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR from entry ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                stoploss_price = entry_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                stoploss_price = entry_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but regime turns bearish
            if position_side > 0 and regime_bear and price_below_kama:
                regime_reversal = True
            # Short position but regime turns bullish
            if position_side < 0 and regime_bull and price_above_kama:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals