#!/usr/bin/env python3
"""
Experiment #552: 12h Primary + 1d/1w HTF — Dual Regime (Chop + Trend) with HTF Bias

Hypothesis: After 490+ failed strategies, the clearest pattern is:
- Single-regime strategies fail in mixed markets (2022 crash + 2025 bear)
- Dual-regime (chop vs trend) adapts to market conditions
- 12h timeframe targets 20-50 trades/year (optimal per Rule 10)
- 1d HTF provides major trend bias (prevents counter-trend disasters)
- Simpler entry logic = more trades = better Sharpe (avoid 0-trade failure)

This strategy uses DUAL REGIME logic:
1. Choppiness Index (14) detects regime: CHOP>61.8=range, CHOP<38.2=trend
2. RANGE regime: Mean revert at Bollinger Band extremes (BB 2.5 std)
3. TREND regime: Follow 12h HMA(21) with RSI(14) pullback entries
4. 1d HMA(50) for major trend bias (only trade with 1d direction)
5. 1w HMA(50) for secular trend filter (avoid major counter-trend)
6. ATR(14) 2.5x trailing stop for all positions
7. Discrete sizing: 0.25 base, 0.30 with HTF confluence

Why this might beat Sharpe=0.435:
- Adapts to both trending (2021, 2023) and ranging (2022, 2025) markets
- 1d/1w HTF prevents major counter-trend losses (key failure in 2022)
- Dual regime = trades in all market conditions (no dead zones)
- 12h TF balances trade frequency vs fee drag (20-50/year target)
- Discrete position sizing (0.0, ±0.25, ±0.30) minimizes fee churn

Position sizing: 0.25-0.30 base (discrete per Rule 4, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_chop_hma_1d1w_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_bollinger_bands(close, period=20, std_dev=2.5):
    """Calculate Bollinger Bands with configurable std dev."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF HMA for major trend direction
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Calculate 1w HTF HMA for secular trend
    hma_1w_50 = calculate_hma(df_1w['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    hma_1w_50_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_50)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    # 12h HMA for trend confirmation
    hma_12h_21 = calculate_hma(close, period=21)
    hma_12h_50 = calculate_hma(close, period=50)
    
    # Bollinger Bands for mean reversion
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_dev=2.5)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_CONFLUENCE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(hma_1w_50_aligned[i]):
            continue
        if np.isnan(hma_12h_21[i]) or np.isnan(hma_12h_50[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime_1d = close[i] > hma_1d_50_aligned[i]
        bear_regime_1d = close[i] < hma_1d_50_aligned[i]
        
        # 1d HMA slope for trend strength
        hma_1d_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_1d_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === 1W SECULAR TREND (avoid major counter-trend) ===
        bull_regime_1w = close[i] > hma_1w_50_aligned[i]
        bear_regime_1w = close[i] < hma_1w_50_aligned[i]
        
        # === 12H TREND CONFIRMATION ===
        bull_regime_12h = close[i] > hma_12h_21[i]
        bear_regime_12h = close[i] < hma_12h_21[i]
        
        hma_12h_slope_bull = hma_12h_21[i] > hma_12h_50[i]
        hma_12h_slope_bear = hma_12h_21[i] < hma_12h_50[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        # CHOP > 61.8 = ranging market (mean revert)
        # CHOP < 38.2 = trending market (trend follow)
        # 38.2 <= CHOP <= 61.8 = transition (no trades or reduced size)
        chop_range = chop_14[i] > 61.8
        chop_trend = chop_14[i] < 38.2
        chop_transition = not chop_range and not chop_trend
        
        # === ENTRY LOGIC — DUAL REGIME ===
        new_signal = 0.0
        
        # LONG ENTRY CONDITIONS
        long_bias = bull_regime_1d and bull_regime_1w
        
        if long_bias:
            # TREND REGIME: HMA pullback + RSI dip
            if chop_trend and bull_regime_12h:
                # RSI pullback to 40-55 in uptrend
                rsi_pullback_long = 40.0 <= rsi_14[i] <= 55.0
                if rsi_pullback_long:
                    if hma_1d_slope_bull:
                        new_signal = POSITION_SIZE_CONFLUENCE
                    else:
                        new_signal = POSITION_SIZE_BASE
            
            # RANGE REGIME: Mean revert at BB lower band
            elif chop_range:
                # Price at/near BB lower + RSI oversold
                at_bb_lower = close[i] <= bb_lower[i] * 1.002  # within 0.2%
                rsi_oversold = rsi_14[i] < 45.0
                if at_bb_lower and rsi_oversold:
                    new_signal = POSITION_SIZE_BASE
        
        # SHORT ENTRY CONDITIONS
        short_bias = bear_regime_1d and bear_regime_1w
        
        if short_bias:
            # TREND REGIME: HMA pullback + RSI rally
            if chop_trend and bear_regime_12h:
                # RSI rally to 45-60 in downtrend
                rsi_pullback_short = 45.0 <= rsi_14[i] <= 60.0
                if rsi_pullback_short:
                    if hma_1d_slope_bear:
                        new_signal = -POSITION_SIZE_CONFLUENCE
                    else:
                        new_signal = -POSITION_SIZE_BASE
            
            # RANGE REGIME: Mean revert at BB upper band
            elif chop_range:
                # Price at/near BB upper + RSI overbought
                at_bb_upper = close[i] >= bb_upper[i] * 0.998  # within 0.2%
                rsi_overbought = rsi_14[i] > 55.0
                if at_bb_upper and rsi_overbought:
                    new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
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
        
        # === EXIT CONDITIONS (regime flip) ===
        # Exit long on 1d regime flip to bear
        if in_position and position_side > 0:
            if bear_regime_1d and hma_1d_slope_bear:
                new_signal = 0.0
            elif bear_regime_12h and hma_12h_slope_bear:
                new_signal = 0.0
        
        # Exit short on 1d regime flip to bull
        if in_position and position_side < 0:
            if bull_regime_1d and hma_1d_slope_bull:
                new_signal = 0.0
            elif bull_regime_12h and hma_12h_slope_bull:
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
                # Flip position
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