#!/usr/bin/env python3
"""
Experiment #339: 4h Primary + 1d HTF — Choppiness Regime + HMA Trend + RSI Entries

Hypothesis: After 30+ failed 4h experiments, the key insight is regime detection.
Most 4h strategies fail because they apply trend logic in choppy markets and
mean-reversion in trending markets. This strategy uses Choppiness Index to
switch between two modes:

1. CHOP > 61.8 (choppy/range): Mean-reversion at Bollinger bands with RSI extremes
2. CHOP < 38.2 (trending): Trend-following with HMA crossover + RSI pullback
3. CHOP 38.2-61.8 (transition): Reduced position size, wait for clarity

Why 4h with 1d HTF might work:
- 1d HMA(21) gives major trend direction without 1w lag
- 4h entries capture multi-day swings (20-50 trades/year target)
- Choppiness filter prevents whipsaw in range markets (2022 crash pattern)
- Asymmetric sizing favors longs (crypto bias) but allows shorts in bear

Key differences from failed experiments:
- NO complex multi-condition AND gates (caused 0 trades in #328, #330, #331)
- Loose RSI thresholds (30-70 not 20-80) to generate trades
- Frequency safeguard every 20 bars if no signal
- 1d HTF trend filter (not 1w - too slow for 4h entries)

Position sizing: 0.25-0.30 longs, 0.15-0.20 shorts
Stoploss: 2.5 * ATR trailing
Target: 30-60 trades/year on 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_hma_rsi_1d_v1"
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

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Much less lag than EMA while maintaining smoothness.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8: Choppy/Range market (mean-reversion favored)
    - CHOP < 38.2: Trending market (trend-following favored)
    - 38.2 < CHOP < 61.8: Transition zone
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    chop[np.isnan(chop)] = 50.0
    
    return chop

def calculate_bollinger(close, period=20, std_mult=2.0):
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
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_14 = calculate_rsi(close, 14)
    hma_4h_8 = calculate_hma(close, period=8)
    hma_4h_21 = calculate_hma(close, period=21)
    chop_14 = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.15
    SHORT_STRONG = 0.20
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(hma_4h_8[i]) or np.isnan(hma_4h_21[i]):
            continue
        
        # === 1D MAJOR TREND REGIME ===
        # Bull: price above 1d HMA (favor longs)
        # Bear: price below 1d HMA (allow shorts)
        regime_bull = close[i] > hma_1d_21_aligned[i]
        regime_bear = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME (mode switch) ===
        chop_val = chop_14[i]
        choppy_market = chop_val > 61.8
        trending_market = chop_val < 38.2
        transition_market = not choppy_market and not trending_market
        
        # Volatility scaling
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        high_vol = atr_ratio > 1.5
        vol_scale = 0.7 if high_vol else 1.0
        
        # === 4H LOCAL TREND ===
        hma_bullish = hma_4h_8[i] > hma_4h_21[i]
        hma_bearish = hma_4h_8[i] < hma_4h_21[i]
        
        # HMA slope
        hma_slope_up = hma_4h_21[i] > hma_4h_21[i-2] if i >= 2 else False
        hma_slope_down = hma_4h_21[i] < hma_4h_4h_21[i-2] if i >= 2 else False
        
        # Price position
        price_above_hma = close[i] > hma_4h_21[i]
        price_below_hma = close[i] < hma_4h_21[i]
        
        # Bollinger position
        price_near_bb_lower = close[i] < bb_lower[i] * 1.01 if not np.isnan(bb_lower[i]) else False
        price_near_bb_upper = close[i] > bb_upper[i] * 0.99 if not np.isnan(bb_upper[i]) else False
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_neutral = 40.0 <= rsi_14[i] <= 60.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # MODE 1: TRENDING MARKET (CHOP < 38.2) - Trend following
        if trending_market:
            if regime_bull:
                # Long: HMA bullish + RSI not overbought + price above HMA
                if hma_bullish and not rsi_overbought and price_above_hma:
                    new_signal = LONG_BASE * vol_scale
                # Strong long: HMA crossover up + RSI rising
                elif hma_bullish and hma_slope_up and rsi_rising:
                    if new_signal == 0.0:
                        new_signal = LONG_STRONG * vol_scale
            
            if regime_bear:
                # Short: HMA bearish + RSI not oversold + price below HMA
                if hma_bearish and not rsi_oversold and price_below_hma:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE * vol_scale
                # Strong short: HMA crossover down + RSI falling
                elif hma_bearish and hma_slope_down and rsi_falling:
                    if new_signal == 0.0:
                        new_signal = -SHORT_STRONG * vol_scale
        
        # MODE 2: CHOPPY MARKET (CHOP > 61.8) - Mean reversion
        elif choppy_market:
            if regime_bull:
                # Long at BB lower + RSI oversold
                if price_near_bb_lower and rsi_oversold:
                    new_signal = LONG_BASE * vol_scale
                # Long at RSI extreme oversold
                elif rsi_14[i] < 30.0:
                    if new_signal == 0.0:
                        new_signal = LONG_STRONG * vol_scale
            
            if regime_bear:
                # Short at BB upper + RSI overbought
                if price_near_bb_upper and rsi_overbought:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE * vol_scale
                # Short at RSI extreme overbought
                elif rsi_14[i] > 70.0:
                    if new_signal == 0.0:
                        new_signal = -SHORT_STRONG * vol_scale
        
        # MODE 3: TRANSITION (38.2 < CHOP < 61.8) - Reduced size, wait for clarity
        elif transition_market:
            if regime_bull and hma_bullish and rsi_rising and not rsi_overbought:
                new_signal = LONG_BASE * 0.6 * vol_scale
            elif regime_bear and hma_bearish and rsi_falling and not rsi_oversold:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.6 * vol_scale
        
        # === FREQUENCY SAFEGUARD (ensure 30+ trades/year on 4h) ===
        # Force trade if no signal for 20 bars (~80 hours = 3.3 days)
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 35.0 and not rsi_overbought:
                new_signal = LONG_BASE * 0.5 * vol_scale
            elif regime_bear and rsi_14[i] < 65.0 and not rsi_oversold:
                new_signal = -SHORT_BASE * 0.5 * vol_scale
            elif rsi_14[i] < 35.0:
                new_signal = LONG_BASE * 0.5 * vol_scale
            elif rsi_14[i] > 65.0:
                new_signal = -SHORT_BASE * 0.5 * vol_scale
        
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
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_14[i] > 70.0:
                rsi_exit = True
            if position_side < 0 and rsi_14[i] < 30.0:
                rsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and regime_bear and price_below_hma:
                regime_reversal = True
            if position_side < 0 and regime_bull and price_above_hma:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG * vol_scale
            elif new_signal > 0:
                new_signal = LONG_BASE * vol_scale
            elif new_signal < -0.18:
                new_signal = -SHORT_STRONG * vol_scale
            else:
                new_signal = -SHORT_BASE * vol_scale
        
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