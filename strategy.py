#!/usr/bin/env python3
"""
Experiment #539: 4h Primary + 1d HTF — Dual Regime (Trend + Mean Revert)

Hypothesis: After 480+ failed strategies, the key insight is that markets alternate
between trending and ranging regimes. Using Choppiness Index to detect regime and
switching logic accordingly should improve Sharpe across all market conditions.

Key insights from failures:
- Pure trend-following fails in 2022 crash and 2025 bear market
- Pure mean-reversion fails in strong trends (2021 bull run)
- Complex volspike/Fisher combos = 0 trades (too many filters)
- 4h TF with 1d filter worked well in exp#529 (Sharpe=0.169)

This strategy uses:
1. 1d HMA(21) for major trend direction (HTF filter)
2. Choppiness Index(14) for regime detection (>61.8=range, <38.2=trend)
3. Trend regime: Donchian(20) breakout + HMA confirmation
4. Range regime: RSI(14) extremes + Bollinger mean reversion
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete position sizing (0.28) to minimize fee churn

Why this might work:
- Regime detection adapts to market conditions (bull/bear/range)
- 1d trend filter prevents counter-trend trades in strong trends
- Donchian breakout captures momentum in trending markets
- RSI+BB mean reversion captures reversals in choppy markets
- 4h TF targets 20-50 trades/year (optimal fee/trade ratio)
- Simpler logic than failed volspike/Fisher strategies

Position sizing: 0.28 (discrete, max 0.40 per rules)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_chop_donchian_1d_v1"
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

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper.values, lower.values, sma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = Choppy/Range market
    CHOP < 38.2 = Trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high > lowest_low and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF HMA for major trend direction
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    # Calculate 4h HMA for trend confirmation
    hma_4h_21 = calculate_hma(close, 21)
    hma_4h_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]) or np.isnan(bb_upper[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(donch_upper[i]) or np.isnan(hma_4h_21[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength confirmation
        hma_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === CHOPPY INDEX REGIME DETECTION ===
        choppy_regime = chop_14[i] > 55.0  # Range/choppy market
        trending_regime = chop_14[i] < 45.0  # Trending market
        # Neutral zone: 45-55 (use trend-following by default)
        
        # === 4H TREND CONFIRMATION ===
        hma_4h_bull = hma_4h_21[i] > hma_4h_50[i]
        hma_4h_bear = hma_4h_21[i] < hma_4h_50[i]
        price_above_hma4h = close[i] > hma_4h_21[i]
        price_below_hma4h = close[i] < hma_4h_21[i]
        
        # === ENTRY LOGIC — DUAL REGIME ===
        new_signal = 0.0
        
        # TRENDING REGIME (Donchian breakout + HMA confirmation)
        if trending_regime or (not choppy_regime and not trending_regime):
            # LONG: Donchian breakout + bull regime + 4h HMA bull
            if bull_regime and hma_4h_bull and close[i] > donch_upper[i-1]:
                new_signal = POSITION_SIZE
            # SHORT: Donchian breakdown + bear regime + 4h HMA bear
            elif bear_regime and hma_4h_bear and close[i] < donch_lower[i-1]:
                new_signal = -POSITION_SIZE
        
        # CHOPPY/RANGE REGIME (RSI + Bollinger mean reversion)
        if choppy_regime:
            # LONG: RSI oversold + price near BB lower + bull regime preferred
            if rsi_14[i] < 35.0 and close[i] < bb_lower[i] * 1.002:
                if bull_regime:
                    new_signal = POSITION_SIZE
                else:
                    new_signal = POSITION_SIZE * 0.6  # smaller size counter-trend
            # SHORT: RSI overbought + price near BB upper + bear regime preferred
            elif rsi_14[i] > 65.0 and close[i] > bb_upper[i] * 0.998:
                if bear_regime:
                    new_signal = -POSITION_SIZE
                else:
                    new_signal = -POSITION_SIZE * 0.6  # smaller size counter-trend
        
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
        # Exit long on regime flip to strong bear
        if in_position and position_side > 0:
            if bear_regime and hma_slope_bear and chop_14[i] < 40.0:
                new_signal = 0.0
        
        # Exit short on regime flip to strong bull
        if in_position and position_side < 0:
            if bull_regime and hma_slope_bull and chop_14[i] < 40.0:
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