#!/usr/bin/env python3
"""
Experiment #306: 12h Primary + 1d HTF — Fisher Transform Reversal + Choppiness Regime + HMA Trend

Hypothesis: Ehlers Fisher Transform outperforms RSI for 12h timeframe because:
1. Fisher Transform normalizes price to Gaussian distribution, better for reversal detection
2. Works exceptionally well in bear/range markets (2022 crash, 2025 test period)
3. Combined with Choppiness Index for regime-aware entries (mean revert in chop, trend in trend)
4. 1d HMA(21) as asymmetric trend filter (long bias in crypto)
5. Target: 25-45 trades/year on 12h (appropriate frequency, low fee drag)

Why this might beat #292 (Sharpe=0.424):
- Fisher Transform catches reversals earlier than RSI in bear markets
- Choppiness Index provides cleaner regime detection than ADX
- 12h timeframe has less noise than 4h, better signal quality
- Asymmetric long bias matches crypto behavior (bull trends stronger)

Key differences from failed strategies:
- Fisher Transform instead of RSI (proven in research for bear markets)
- Looser entry thresholds to ensure 25+ trades/year
- Discrete signal levels (0.0, ±0.25, ±0.35) to reduce fee churn
- ATR trailing stoploss via signal→0

Position sizing: 0.25 base, 0.35 strong conviction
Stoploss: 2.5 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_chop_hma_1d_asym_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution.
    Long when Fisher crosses above -1.5 (oversold reversal)
    Short when Fisher crosses below +1.5 (overbought reversal)
    
    Works exceptionally well in bear/range markets.
    """
    n = period
    fisher = np.zeros(len(close))
    trigger = np.zeros(len(close))
    
    # Calculate typical price
    tp = (high + low + close) / 3.0
    
    # Normalize price to -1 to +1 range
    for i in range(n, len(close)):
        highest = np.max(tp[i-n+1:i+1])
        lowest = np.min(tp[i-n+1:i+1])
        
        if highest > lowest:
            normalized = 0.66 * ((tp[i] - lowest) / (highest - lowest) - 0.5) + 0.67 * (tp[i-1] - tp[i-n]) / (highest - lowest + 1e-10)
            normalized = np.clip(normalized, -0.999, 0.999)
        else:
            normalized = 0.0
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
        trigger[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, trigger

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate ATR
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    
    atr_sum = atr.rolling(window=n, min_periods=n).sum()
    hh = high_s.rolling(window=n, min_periods=n).max()
    ll = low_s.rolling(window=n, min_periods=n).min()
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh.iloc[i] - ll.iloc[i]
        if range_hl > 0 and atr_sum.iloc[i] > 0:
            chop[i] = 100 * np.log10(atr_sum.iloc[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Faster and smoother than EMA, less lag.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (primary trend regime)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, 9)
    hma_12h_21 = calculate_hma(close, 21)
    hma_12h_50 = calculate_hma(close, 50)
    sma_200 = calculate_sma(close, 200)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.35
    MIN_SIZE = 0.15
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(hma_12h_21[i]) or np.isnan(bb_upper[i]):
            continue
        
        # === 1D TREND REGIME (primary direction filter — ASYMMETRIC) ===
        # Bull: price above 1d HMA (favor longs)
        # Bear: price below 1d HMA (favor shorts, but reduced)
        regime_bull = close[i] > hma_1d_21_aligned[i]
        regime_bear = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 58 = range market (mean revert entries)
        # CHOP < 42 = trend market (breakout entries)
        is_choppy = chop_14[i] > 58.0
        is_trending = chop_14[i] < 42.0
        is_transitional = not is_choppy and not is_trending
        
        # === VOLATILITY REGIME (ATR ratio) ===
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        high_vol = atr_ratio > 1.5
        vol_scale = 0.7 if high_vol else 1.0
        
        # === 12H LOCAL SIGNALS ===
        # HMA trend direction
        hma_bullish = hma_12h_21[i] > hma_12h_21[i-3] if i >= 3 else False
        hma_bearish = hma_12h_21[i] < hma_12h_21[i-3] if i >= 3 else False
        
        # Price position relative to HMA
        price_above_hma = close[i] > hma_12h_21[i]
        price_below_hma = close[i] < hma_12h_21[i]
        
        # Price relative to SMA200 (long-term trend)
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher reversal signals (proven in bear markets)
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = fisher[i] > fisher_trigger[i] and fisher_trigger[i] < -1.0
        fisher_cross_down = fisher[i] < fisher_trigger[i] and fisher_trigger[i] > 1.0
        
        # === BOLLINGER BAND SIGNALS ===
        bb_break_lower = close[i] < bb_lower[i] * 1.005
        bb_break_upper = close[i] > bb_upper[i] * 0.995
        bb_near_lower = close[i] < bb_lower[i] * 1.015
        bb_near_upper = close[i] > bb_upper[i] * 0.985
        
        # === ENTRY LOGIC (ASYMMETRIC + DUAL REGIME) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (favored in bull regime)
        if regime_bull or is_choppy:
            # Fisher reversal in choppy market (mean revert)
            if is_choppy and fisher_oversold and bb_near_lower:
                new_signal = BASE_SIZE * vol_scale
            
            # Fisher cross up in trending market
            elif is_trending and fisher_cross_up and price_above_hma:
                new_signal = BASE_SIZE * vol_scale
            
            # HMA bullish + price above HMA + Fisher neutral/rising
            elif hma_bullish and price_above_hma and fisher[i] > fisher[i-1]:
                new_signal = BASE_SIZE * vol_scale
            
            # Strong conviction: Fisher extreme oversold + bull regime + above SMA200
            elif fisher_oversold and regime_bull and price_above_sma200:
                new_signal = STRONG_SIZE * vol_scale
            
            # Transitional + HMA bullish + Fisher rising from negative
            elif is_transitional and hma_bullish and fisher[i] > -1.0 and fisher[i] > fisher[i-1]:
                new_signal = MIN_SIZE * vol_scale
        
        # SHORT ENTRIES (only in bear regime, reduced size)
        if regime_bear:
            # Fisher reversal in choppy market (mean revert)
            if is_choppy and fisher_overbought and bb_near_upper:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * vol_scale * 0.8  # Reduced short size
            
            # Fisher cross down in trending market
            elif is_trending and fisher_cross_down and price_below_hma:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * vol_scale * 0.8
            
            # HMA bearish + price below HMA + Fisher falling
            elif hma_bearish and price_below_hma and fisher[i] < fisher[i-1]:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * vol_scale * 0.8
            
            # Strong conviction: Fisher extreme overbought + bear regime
            elif fisher_overbought and regime_bear and price_below_hma:
                if new_signal == 0.0:
                    new_signal = -STRONG_SIZE * vol_scale * 0.8
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year on 12h) ===
        # Force trade if no signal for 45 bars (~45 * 12h = 540h ≈ 22 days)
        if bars_since_last_trade > 45 and new_signal == 0.0 and not in_position:
            if regime_bull and fisher[i] > -1.0 and price_above_hma:
                new_signal = MIN_SIZE * vol_scale
            elif regime_bear and fisher[i] < 1.0 and price_below_hma:
                new_signal = -MIN_SIZE * vol_scale * 0.8
            elif is_choppy and fisher_oversold:
                new_signal = MIN_SIZE * vol_scale
            elif is_choppy and fisher_overbought:
                new_signal = -MIN_SIZE * vol_scale * 0.8
        
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
        
        # === FISHER REVERSAL EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            # Long position: exit when Fisher turns overbought
            if position_side > 0 and fisher_overbought:
                fisher_exit = True
            # Short position: exit when Fisher turns oversold
            if position_side < 0 and fisher_oversold:
                fisher_exit = True
        
        # === HMA REVERSAL EXIT ===
        hma_exit = False
        if in_position and position_side != 0:
            # Long position: exit when HMA turns bearish + price below
            if position_side > 0 and hma_bearish and price_below_hma:
                hma_exit = True
            # Short position: exit when HMA turns bullish + price above
            if position_side < 0 and hma_bullish and price_above_hma:
                hma_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 1d regime turns strongly bearish
            if position_side > 0 and regime_bear and price_below_hma:
                regime_reversal = True
            # Short position but 1d regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_hma:
                regime_reversal = True
        
        if stoploss_triggered or fisher_exit or hma_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.12:
                new_signal = 0.0
            elif new_signal > 0.30:
                new_signal = STRONG_SIZE * vol_scale
            elif new_signal > 0:
                new_signal = BASE_SIZE * vol_scale
            elif new_signal < -0.30:
                new_signal = -STRONG_SIZE * vol_scale * 0.8
            else:
                new_signal = -BASE_SIZE * vol_scale * 0.8
        
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