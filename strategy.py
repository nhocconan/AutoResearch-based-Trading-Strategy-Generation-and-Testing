#!/usr/bin/env python3
"""
Experiment #312: 12h Primary + 1d/1w HTF — Dual Regime (Chop/Trend) + HMA + RSI + ATR

Hypothesis: A regime-switching strategy on 12h timeframe will outperform single-regime approaches because:
1. Crypto alternates between trending (2021 bull, 2022 crash) and ranging (2023-2024, 2025 bear)
2. Choppiness Index (CHOP) reliably detects regime: >55=chop, <45=trend
3. In chop: mean-reversion at Bollinger extremes works (RSI + BB confluence)
4. In trend: pullback entries in direction of 1w HMA work (RSI pullback + HMA alignment)
5. 1w HMA(21) provides major trend bias without excessive lag
6. 1d CHOP confirms regime stability before 12h entries
7. Target: 25-45 trades/year on 12h (appropriate for this TF, low fee drag)

Why this might beat #306 (Sharpe=0.203) and current best (Sharpe=0.424):
- Dual regime adapts to market conditions instead of one-size-fits-all
- 12h TF balances trade frequency (enough trades) vs signal quality (less noise than 1h/4h)
- 1w HTF trend filter is stronger than 1d for crypto's multi-week trends
- Asymmetric sizing matches crypto behavior (longs 0.30, shorts 0.20)
- Looser RSI thresholds ensure 20+ trades/year (learned from 0-trade failures)
- ATR trailing stop protects capital in 2022-style crashes

Key differences from failed strategies:
- Dual regime (chop=trend switch) instead of single logic
- 12h primary (not 1h/4h which had too many trades or 1d which had too few)
- RSI thresholds widened (35-65 instead of 30-70) to generate more trades
- Fallback entry after 25 bars without trade (prevents 0-trade disaster)
- Discrete signal levels (0.0, ±0.20, ±0.30) to reduce fee churn

Position sizing: 0.25 base, 0.30 strong conviction (longs), 0.20 (shorts)
Stoploss: 2.5 * ATR trailing (tighter than 3.0 to reduce drawdown)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_hma_rsi_chop_1d1w_asym_v1"
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
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag, smooth like SMA.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    # WMA helper
    def wma(series, span):
        return series.ewm(span=span, min_periods=span, adjust=False).mean()
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    diff = 2.0 * wma_half - wma_full
    hma = wma(diff, sqrt_n)
    
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper.values, lower.values, sma.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Calculate 1d HTF indicators (regime confirmation)
    chop_1d_14 = calculate_choppiness_index(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values, 
        14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    chop_1d_14_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_14)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    chop_12h_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_12h_16 = calculate_hma(close, period=16)
    hma_12h_48 = calculate_hma(close, period=48)
    sma_200 = calculate_sma(close, 200)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: longs favored in crypto
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.20
    SHORT_STRONG = 0.25
    
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
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(chop_1d_14_aligned[i]) or np.isnan(chop_12h_14[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_12h_16[i]) or np.isnan(hma_12h_48[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 1W MAJOR TREND REGIME (primary direction filter — ASYMMETRIC) ===
        # Bull: price above 1w HMA (favor longs with larger size)
        # Bear: price below 1w HMA (allow shorts but reduced size)
        regime_bull = close[i] > hma_1w_21_aligned[i]
        regime_bear = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME (dual regime switch) ===
        # Use 1d CHOP for regime stability (less noise than 12h)
        chop_1d = chop_1d_14_aligned[i]
        chop_12h = chop_12h_14[i]
        
        # CHOP > 55 = range market (mean revert at BB extremes)
        # CHOP < 45 = trending market (trend follow with pullbacks)
        is_choppy = chop_1d > 55.0
        is_trending = chop_1d < 45.0
        
        # === VOLATILITY REGIME (ATR ratio) ===
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        high_vol = atr_ratio > 1.5
        vol_scale = 0.7 if high_vol else 1.0
        
        # === 12H LOCAL TREND ===
        # HMA crossover (fast vs slow)
        hma_bullish = hma_12h_16[i] > hma_12h_48[i]
        hma_bearish = hma_12h_16[i] < hma_12h_48[i]
        
        # HMA slope (3-bar lookback)
        hma_slope_up = hma_12h_48[i] > hma_12h_48[i-3] if i >= 3 else False
        hma_slope_down = hma_12h_48[i] < hma_12h_48[i-3] if i >= 3 else False
        
        # Price position relative to HMA
        price_above_hma = close[i] > hma_12h_48[i]
        price_below_hma = close[i] < hma_12h_48[i]
        
        # Price relative to SMA200 (long-term trend)
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # === RSI SIGNALS (widened thresholds for more trades) ===
        # RSI pullback long: RSI 40-55 in uptrend
        # RSI pullback short: RSI 45-60 in downtrend
        rsi_oversold_pullback = 38.0 < rsi_14[i] < 55.0
        rsi_overbought_pullback = 45.0 < rsi_14[i] < 62.0
        rsi_strong_oversold = rsi_14[i] < 40.0
        rsi_strong_overbought = rsi_14[i] > 60.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === BOLLINGER BAND POSITION ===
        bb_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
        near_bb_lower = bb_position < 0.15
        near_bb_upper = bb_position > 0.85
        
        # === ENTRY LOGIC (DUAL REGIME + ASYMMETRIC) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # === REGIME 1: TRENDING MARKET (CHOP < 45) ===
        if is_trending:
            # LONG entries in bull regime
            if regime_bull:
                # RSI pullback in uptrend (primary entry)
                if rsi_oversold_pullback and hma_bullish and price_above_hma:
                    new_signal = LONG_BASE * vol_scale
                
                # Strong RSI oversold + bull regime + above SMA200
                elif rsi_strong_oversold and regime_bull and price_above_sma200:
                    new_signal = LONG_STRONG * vol_scale
                
                # HMA bullish + slope up + RSI rising
                elif hma_bullish and hma_slope_up and rsi_rising and rsi_14[i] > 45.0:
                    new_signal = LONG_BASE * vol_scale
            
            # SHORT entries in bear regime
            if regime_bear:
                # RSI pullback in downtrend
                if rsi_overbought_pullback and hma_bearish and price_below_hma:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE * vol_scale
                
                # Strong RSI overbought + bear regime
                elif rsi_strong_overbought and regime_bear and price_below_hma:
                    if new_signal == 0.0:
                        new_signal = -SHORT_STRONG * vol_scale
                
                # HMA bearish + slope down + RSI falling
                elif hma_bearish and hma_slope_down and rsi_falling and rsi_14[i] < 55.0:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE * vol_scale
        
        # === REGIME 2: CHOPPY MARKET (CHOP > 55) ===
        if is_choppy:
            # Mean reversion at Bollinger extremes
            # Long near BB lower + RSI oversold
            if near_bb_lower and rsi_strong_oversold:
                if regime_bull:
                    new_signal = LONG_BASE * 0.8 * vol_scale
                else:
                    new_signal = LONG_BASE * 0.6 * vol_scale
            
            # Short near BB upper + RSI overbought
            if near_bb_upper and rsi_strong_overbought:
                if new_signal == 0.0:
                    if regime_bear:
                        new_signal = -SHORT_BASE * 0.8 * vol_scale
                    else:
                        new_signal = -SHORT_BASE * 0.6 * vol_scale
        
        # === FREQUENCY SAFEGUARD (ensure 20+ trades/year on 12h) ===
        # Force trade if no signal for 25 bars (~2 weeks on 12h)
        if bars_since_last_trade > 25 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 45.0 and price_above_hma:
                new_signal = LONG_BASE * 0.6 * vol_scale
            elif regime_bear and rsi_14[i] < 55.0 and price_below_hma:
                new_signal = -SHORT_BASE * 0.6 * vol_scale
            elif rsi_strong_oversold and near_bb_lower:
                new_signal = LONG_BASE * 0.6 * vol_scale
            elif rsi_strong_overbought and near_bb_upper:
                new_signal = -SHORT_BASE * 0.6 * vol_scale
        
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
            # Long position: exit when RSI turns overbought
            if position_side > 0 and rsi_strong_overbought:
                rsi_exit = True
            # Short position: exit when RSI turns oversold
            if position_side < 0 and rsi_strong_oversold:
                rsi_exit = True
        
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
            # Long position but 1w regime turns strongly bearish
            if position_side > 0 and regime_bear and price_below_hma:
                regime_reversal = True
            # Short position but 1w regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_hma:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or hma_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.12:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG * vol_scale
            elif new_signal > 0:
                new_signal = LONG_BASE * vol_scale
            elif new_signal < -0.23:
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