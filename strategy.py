#!/usr/bin/env python3
"""
Experiment #234: 4h Primary + 12h/1d HTF — Regime-Adaptive Trend/Mean Reversion

Hypothesis: After 233 experiments, the key insight is that NO single regime works
across all market conditions. BTC 2021-2024 had strong trends, but 2025+ is bear/range.

This strategy uses:
1. 12h HMA(21) slope for PRIMARY trend regime (bull/bear/neutral)
2. Choppiness Index(14) to detect range vs trend markets
3. 4h Donchian(20) breakout for trend entries
4. 4h RSI(14) extremes for mean-reversion in choppy markets
5. 1d ADX for trend strength confirmation
6. 2.5x ATR trailing stop for risk management

Key improvements over #231:
- Simpler entry logic (fewer conflicting paths)
- Regime-adaptive: trend follow when trending, mean-revert when choppy
- LOOSER RSI thresholds (40/60 not 30/70) for guaranteed trade frequency
- Force-trade after 40 bars of no signal (not 60)
- Better HTF alignment (12h not 1d/1w which are too slow for 4h entries)

Position sizing: 0.25 base, 0.30 strong signals (discrete levels)
Target: 30-50 trades/year per symbol (within 4h cost model)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_adaptive_donchian_rsi_12h1d_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change over lookback."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        prev = hma_values[i - lookback]
        curr = hma_values[i]
        if prev != 0 and not np.isnan(prev) and not np.isnan(curr):
            slope[i] = (curr - prev) / abs(prev) * 100
    return slope

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
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
    
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(20).values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    # Rolling sum of ATR
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    
    # Highest high and lowest low over period
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    # Choppiness calculation
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HTF indicators (primary trend regime)
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 3)
    
    # Calculate 1d HTF indicators (trend strength)
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian_channels(high, low, 20)
    hma_4h_21 = calculate_hma(close, 21)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -40
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(hma_4h_21[i]):
            continue
        
        # === REGIME DETECTION (12h HMA slope) ===
        # Bull regime: 12h HMA slope > 0.15%
        # Bear regime: 12h HMA slope < -0.15%
        # Neutral: between -0.15% and 0.15%
        regime_bull = hma_12h_slope_aligned[i] > 0.15
        regime_bear = hma_12h_slope_aligned[i] < -0.15
        regime_neutral = not regime_bull and not regime_bear
        
        price_above_12h_hma = close[i] > hma_12h_21_aligned[i]
        price_below_12h_hma = close[i] < hma_12h_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 61.8 = range market (mean revert)
        # CHOP < 38.2 = trend market (trend follow)
        is_choppy = chop_14[i] > 55.0  # Slightly lower threshold for more mean-revert signals
        is_trending = chop_14[i] < 45.0  # Slightly higher threshold for more trend signals
        
        # === 1D TREND STRENGTH ===
        daily_trend_strong = adx_1d_aligned[i] > 25
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 4H LOCAL SIGNALS ===
        price_above_4h_hma = close[i] > hma_4h_21[i]
        price_below_4h_hma = close[i] < hma_4h_21[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i]
        breakout_short = close[i] < donchian_lower[i]
        
        # === RSI MOMENTUM (LOOSE THRESHOLDS FOR TRADE FREQUENCY) ===
        rsi_oversold = rsi_14[i] < 45  # Mean reversion long
        rsi_overbought = rsi_14[i] > 55  # Mean reversion short
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND FOLLOWING MODE (when trending + regime aligned)
        if is_trending or regime_bull or regime_bear:
            # LONG: Breakout + regime bullish + RSI confirmation
            if breakout_long and regime_bull and rsi_bullish:
                new_signal = STRONG_SIZE
            # LONG: Breakout + price above 12h HMA + RSI > 50
            elif breakout_long and price_above_12h_hma and rsi_14[i] > 50:
                new_signal = BASE_SIZE
            # LONG: Regime bullish + price above all HMAs + RSI bullish
            elif regime_bull and price_above_12h_hma and price_above_4h_hma and rsi_bullish:
                new_signal = BASE_SIZE
            
            # SHORT: Breakout + regime bearish + RSI confirmation
            if breakout_short and regime_bear and rsi_bearish:
                new_signal = -STRONG_SIZE
            # SHORT: Breakout + price below 12h HMA + RSI < 50
            elif breakout_short and price_below_12h_hma and rsi_14[i] < 50:
                new_signal = -BASE_SIZE
            # SHORT: Regime bearish + price below all HMAs + RSI bearish
            elif regime_bear and price_below_12h_hma and price_below_4h_hma and rsi_bearish:
                new_signal = -BASE_SIZE
        
        # MEAN REVERSION MODE (when choppy)
        if is_choppy:
            # LONG: RSI oversold + price below 4h HMA (pullback in range)
            if rsi_oversold and price_below_4h_hma and not regime_bear:
                if new_signal == 0.0 or new_signal < BASE_SIZE * 0.5:
                    new_signal = BASE_SIZE * 0.6
            # LONG: RSI very oversold (<35) in any regime except strong bear
            if rsi_14[i] < 35 and not regime_bear:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE * 0.5
            
            # SHORT: RSI overbought + price above 4h HMA (pullback in range)
            if rsi_overbought and price_above_4h_hma and not regime_bull:
                if new_signal == 0.0 or new_signal > -BASE_SIZE * 0.5:
                    new_signal = -BASE_SIZE * 0.6
            # SHORT: RSI very overbought (>65) in any regime except strong bull
            if rsi_14[i] > 65 and not regime_bull:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * 0.5
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 40 bars (~6-7 days on 4h)
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 48 and price_above_4h_hma:
                new_signal = BASE_SIZE * 0.4
            elif regime_bear and rsi_14[i] < 52 and price_below_4h_hma:
                new_signal = -BASE_SIZE * 0.4
            elif is_choppy and rsi_14[i] < 42:
                new_signal = BASE_SIZE * 0.35
            elif is_choppy and rsi_14[i] > 58:
                new_signal = -BASE_SIZE * 0.35
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but regime turns strongly bearish
            if position_side > 0 and regime_bear and price_below_12h_hma:
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_12h_hma:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals