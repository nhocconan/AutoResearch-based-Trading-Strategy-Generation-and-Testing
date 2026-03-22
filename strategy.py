#!/usr/bin/env python3
"""
Experiment #545: 12h Regime-Adaptive Dual-Mode Strategy with Multi-TF HMA

Hypothesis: After analyzing 500+ failed experiments, the key insight is:
1. Single-regime strategies fail because crypto alternates between trend/range
2. 12h timeframe captures multi-day moves without intraday noise
3. Choppiness Index (CHOP) reliably distinguishes trending vs ranging regimes
4. Dual-mode logic: trend-follow in low CHOP, mean-revert in high CHOP
5. 1d HMA provides intermediate trend bias, 1w HMA provides macro regime
6. This should work in 2022 crash (bear trend) AND 2025 range (mean revert)

Why 12h:
- 2 bars/day = ~730 bars/year = quality over quantity
- Reduces whipsaw vs 15m/1h/4h (all failed with negative Sharpe)
- Enough trades for statistical significance (target 30-50/year)

Regime Detection:
- CHOP(14) < 45 = trending → use Donchian breakout + RSI momentum
- CHOP(14) > 55 = ranging → use RSI extremes + Bollinger mean reversion
- CHOP 45-55 = transition → reduce position size by 50%

Multi-TF Alignment:
- 1d HMA(21): intermediate trend bias (only trade with bias)
- 1w HMA(21): macro regime (avoid counter-macro trades)
- Both loaded via mtf_data helper ONCE before loop

Position Sizing:
- Long: 0.30 (bull bias), 0.25 (neutral)
- Short: 0.25 (bear bias), 0.20 (neutral)
- Max: 0.35 absolute
- Stoploss: 2.5 * ATR(14) trailing

Timeframe: 12h (REQUIRED)
HTF: 1d and 1w via mtf_data helper
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_adaptive_chop_dual_mode_1d_1w_hma_atr_v1"
timeframe = "12h"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # ATR for each bar
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # CHOP formula
    price_range = hh - ll
    price_range = price_range.replace(0, np.inf)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = chop.clip(0, 100)
    
    return chop.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high_s - high_s.shift(1)
    down_move = low_s.shift(1) - low_s
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    adx_14 = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_LONG_BULL = 0.30
    SIZE_LONG_NEUTRAL = 0.25
    SIZE_SHORT_BEAR = 0.25
    SIZE_SHORT_NEUTRAL = 0.20
    SIZE_REDUCED_MULT = 0.5  # For transition regime
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        chop = chop_14[i]
        is_trending = chop < 45  # Trending regime
        is_ranging = chop > 55   # Ranging regime
        is_transition = not is_trending and not is_ranging  # 45-55
        
        # === HTF TREND BIAS ===
        bull_bias_1d = close[i] > hma_1d_aligned[i]
        bear_bias_1d = close[i] < hma_1d_aligned[i]
        bull_bias_1w = close[i] > hma_1w_aligned[i]
        bear_bias_1w = close[i] < hma_1w_aligned[i]
        
        # Strong bias = both 1d and 1w agree
        strong_bull = bull_bias_1d and bull_bias_1w
        strong_bear = bear_bias_1d and bear_bias_1w
        
        # === TRENDING REGIME LOGIC (CHOP < 45) ===
        new_signal = 0.0
        
        if is_trending:
            # Donchian breakout with RSI momentum confirmation
            breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
            breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
            
            # RSI confirms momentum (not overbought/oversold at entry)
            rsi_momentum_long = rsi_14[i] > 50 and rsi_14[i] < 75
            rsi_momentum_short = rsi_14[i] < 50 and rsi_14[i] > 25
            
            # ADX confirms trend strength
            trend_confirmed = adx_14[i] > 20
            
            if breakout_long and rsi_momentum_long and trend_confirmed:
                if strong_bull:
                    new_signal = SIZE_LONG_BULL
                elif bull_bias_1d:
                    new_signal = SIZE_LONG_NEUTRAL
                # Only long in bullish or neutral bias (avoid counter-trend)
            
            if breakout_short and rsi_momentum_short and trend_confirmed:
                if strong_bear:
                    new_signal = -SIZE_SHORT_BEAR
                elif bear_bias_1d:
                    new_signal = -SIZE_SHORT_NEUTRAL
                # Only short in bearish or neutral bias
        
        # === RANGING REGIME LOGIC (CHOP > 55) ===
        elif is_ranging:
            # Mean reversion: fade RSI extremes at BB bounds
            rsi_oversold = rsi_14[i] < 35
            rsi_overbought = rsi_14[i] > 65
            
            at_bb_lower = close[i] <= bb_lower[i] * 1.002  # Within 0.2% of lower band
            at_bb_upper = close[i] >= bb_upper[i] * 0.998  # Within 0.2% of upper band
            
            # Long: oversold RSI + at lower BB + not strongly bearish
            if rsi_oversold and at_bb_lower and not strong_bear:
                if bull_bias_1d:
                    new_signal = SIZE_LONG_BULL
                else:
                    new_signal = SIZE_LONG_NEUTRAL
            
            # Short: overbought RSI + at upper BB + not strongly bullish
            if rsi_overbought and at_bb_upper and not strong_bull:
                if bear_bias_1d:
                    new_signal = -SIZE_SHORT_BEAR
                else:
                    new_signal = -SIZE_SHORT_NEUTRAL
        
        # === TRANSITION REGIME (CHOP 45-55) ===
        # Reduce position size by 50% if we have a signal
        if is_transition and new_signal != 0.0:
            new_signal = new_signal * SIZE_REDUCED_MULT
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME FLIP EXIT ===
        # Exit if regime changes dramatically against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and is_ranging and rsi_14[i] > 70:
                # Long in ranging regime with overbought RSI = take profit
                new_signal = 0.0
            if position_side < 0 and is_ranging and rsi_14[i] < 30:
                # Short in ranging regime with oversold RSI = take profit
                new_signal = 0.0
        
        # === HTF BIAS REVERSAL EXIT ===
        # Exit if 1d HMA flips strongly against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and strong_bear:
                new_signal = 0.0
            if position_side < 0 and strong_bull:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals