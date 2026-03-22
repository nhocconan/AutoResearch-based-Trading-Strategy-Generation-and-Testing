#!/usr/bin/env python3
"""
Experiment #539: 12h Regime-Adaptive Strategy with Fisher Transform + Choppiness Index

Hypothesis: After 500+ failed experiments, the key insight is that NO SINGLE strategy
works across all market regimes. 2021-2024 had strong trends, 2025+ is choppy/bear.

This strategy ADAPTS to regime using Choppiness Index:
- CHOP < 38.2 (trending): Donchian breakout + 1d HMA bias (trend following)
- CHOP > 61.8 (choppy): Fisher Transform extremes (mean reversion)
- CHOP 38.2-61.8 (neutral): Stay flat or reduce exposure

Why this should work on 12h:
1. 12h reduces noise vs intraday timeframes (15m/1h/4h all failed)
2. Fisher Transform catches reversals in bear/range markets (2025+)
3. Choppiness Index prevents trend strategies in chop (major failure mode)
4. 1d HMA bias prevents counter-trend entries in strong trends (2021-2022)
5. Asymmetric logic = works in BOTH bull and bear regimes

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete (max 0.40)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_adaptive_fisher_chop_donchian_daily_hma_atr_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate ATR
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    highest_high = high_s.rolling(window=period, min_periods=period).max()
    lowest_low = low_s.rolling(window=period, min_periods=period).min()
    price_range = highest_high - lowest_low
    
    # Choppiness Index
    chop = 100 * np.log10(atr_sum / price_range.replace(0, np.inf)) / np.log10(period)
    chop = chop.clip(0, 100)
    
    return chop.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution.
    Long when Fisher crosses above -1.5 from below.
    Short when Fisher crosses below +1.5 from above.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Typical price
    typical = (high_s + low_s) / 2
    
    # Normalize to -1 to +1 range
    highest = typical.rolling(window=period, min_periods=period).max()
    lowest = typical.rolling(window=period, min_periods=period).min()
    range_val = highest - lowest
    range_val = range_val.replace(0, np.inf)
    
    normalized = 2 * ((typical - lowest) / range_val) - 1
    normalized = normalized.clip(-0.99, 0.99)  # Prevent log(0)
    
    # Fisher Transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    fisher_prev = fisher.shift(1)
    
    return fisher.values, fisher_prev.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, 9)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trending = chop_14[i] < 38.2
        is_choppy = chop_14[i] > 61.8
        # is_neutral = 38.2 <= chop <= 61.8 (stay flat)
        
        # === 1D HMA TREND BIAS ===
        bull_bias = close[i] > hma_1d_aligned[i]
        bear_bias = close[i] < hma_1d_aligned[i]
        
        # === TREND REGIME: Donchian Breakout ===
        breakout_long = False
        breakout_short = False
        
        if is_trending:
            if not np.isnan(donchian_upper[i-1]):
                breakout_long = close[i] > donchian_upper[i-1]
            if not np.isnan(donchian_lower[i-1]):
                breakout_short = close[i] < donchian_lower[i-1]
        
        # === CHOPPY REGIME: Fisher Transform Mean Reversion ===
        fisher_long = False
        fisher_short = False
        
        if is_choppy:
            # Long: Fisher crosses above -1.5 from below + RSI oversold
            fisher_long = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5) and (rsi_14[i] < 40)
            # Short: Fisher crosses below +1.5 from above + RSI overbought
            fisher_short = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5) and (rsi_14[i] > 60)
        
        # === ENTRY LOGIC (Regime Adaptive) ===
        new_signal = 0.0
        
        # TREND REGIME: Breakout + HTF bias + ADX confirmation
        if is_trending:
            if breakout_long and bull_bias and adx_14[i] > 20:
                new_signal = SIZE
            elif breakout_short and bear_bias and adx_14[i] > 20:
                new_signal = -SIZE
        
        # CHOPPY REGIME: Fisher mean reversion (no HTF bias needed)
        elif is_choppy:
            if fisher_long:
                new_signal = SIZE
            elif fisher_short:
                new_signal = -SIZE
        
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
        
        # === TREND REVERSAL EXIT (Trend Regime Only) ===
        if in_position and new_signal != 0.0 and is_trending:
            if position_side > 0 and bear_bias:
                new_signal = 0.0
            if position_side < 0 and bull_bias:
                new_signal = 0.0
        
        # === CHOPPY REGIME EXIT ===
        # Exit mean reversion trades when Fisher crosses back through 0
        if in_position and new_signal != 0.0 and is_choppy:
            if position_side > 0 and fisher[i] > 0.5:
                new_signal = 0.0
            if position_side < 0 and fisher[i] < -0.5:
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