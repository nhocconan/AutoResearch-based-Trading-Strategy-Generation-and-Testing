#!/usr/bin/env python3
"""
Experiment #524: 4h Primary + 12h/1d HTF — Choppiness Regime + Dual Logic

Hypothesis: After 469 failed strategies, the key insight is REGIME DETECTION.
Research shows Choppiness Index regime switch + Connors RSI gave ETH Sharpe +0.923.

This strategy uses:
1. Choppiness Index (14) to detect range vs trend regime
   - CHOP > 61.8 = range (use mean reversion logic)
   - CHOP < 38.2 = trend (use trend following logic)
   - Between = no trade (avoid whipsaw)
2. 12h HMA(21) for major trend direction filter
3. Regime-specific entries:
   - Range: RSI(14) extremes ( <30 long, >70 short) + BB mean reversion
   - Trend: HMA(16/48) crossover + Donchian(20) breakout confirmation
4. ATR(14) 2.5x trailing stop for risk management
5. Discrete position sizing (0.30) to minimize fee churn

Why this might work:
- Regime detection prevents using wrong logic (mean revert in trend = loss)
- 12h trend filter aligns with major direction (proven in #521)
- Dual logic adapts to market conditions (range 2025 vs trend 2021)
- Simple conditions ensure trade frequency (>30 trades/symbol train)
- 4h TF targets 25-40 trades/year (optimal fee/trade ratio)

Position sizing: 0.30 (discrete, max 0.40 per rules)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_dual_hma_12h_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8 = Range/Consolidation (mean reversion favorable)
    - CHOP < 38.2 = Trend (trend following favorable)
    - 38.2 <= CHOP <= 61.8 = Transition (avoid trading)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    tr_series = pd.Series(tr)
    atr_sum = tr_series.rolling(window=period, min_periods=period).sum().values
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100.0 * np.log10(atr_sum / price_range + 1e-10) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HTF HMA for major trend direction
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_50 = calculate_hma(df_12h['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Choppiness Index for regime detection
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    # HMA for trend following (16/48 crossover)
    hma_4h_16 = calculate_hma(close, period=16)
    hma_4h_48 = calculate_hma(close, period=48)
    
    # RSI for mean reversion entries
    rsi_14 = calculate_rsi(close, 14)
    
    # Bollinger Bands for mean reversion
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # Donchian Channel for breakout confirmation
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track HMA crossover
    prev_hma_16 = np.zeros(n)
    prev_hma_48 = np.zeros(n)
    prev_hma_16[1:] = hma_4h_16[:-1]
    prev_hma_48[1:] = hma_4h_48[:-1]
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_50_aligned[i]):
            continue
        if np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        range_regime = chop_14[i] > 61.8  # Mean reversion favorable
        trend_regime = chop_14[i] < 38.2  # Trend following favorable
        transition_regime = not range_regime and not trend_regime  # Avoid trading
        
        # === 12H MAJOR TREND (primary direction filter) ===
        bull_trend = close[i] > hma_12h_21_aligned[i]
        bear_trend = close[i] < hma_12h_21_aligned[i]
        
        # 12h HMA slope for trend strength
        hma_slope_bull = hma_12h_21_aligned[i] > hma_12h_50_aligned[i]
        hma_slope_bear = hma_12h_21_aligned[i] < hma_12h_50_aligned[i]
        
        # === ENTRY LOGIC — REGIME-SPECIFIC ===
        new_signal = 0.0
        
        # --- RANGE REGIME: Mean Reversion Logic ---
        if range_regime and not transition_regime:
            # Long: RSI oversold + price at/near BB lower + bull trend bias
            if rsi_14[i] < 30.0 and close[i] <= bb_lower[i] * 1.005:
                if bull_trend or not bear_trend:  # Prefer bull, allow neutral
                    new_signal = POSITION_SIZE
            
            # Short: RSI overbought + price at/near BB upper + bear trend bias
            if new_signal == 0.0:
                if rsi_14[i] > 70.0 and close[i] >= bb_upper[i] * 0.995:
                    if bear_trend or not bull_trend:  # Prefer bear, allow neutral
                        new_signal = -POSITION_SIZE
        
        # --- TREND REGIME: Trend Following Logic ---
        elif trend_regime and not transition_regime:
            # HMA crossover signals
            hma_cross_up = (hma_4h_16[i] > hma_4h_48[i]) and (prev_hma_16[i] <= prev_hma_48[i])
            hma_cross_down = (hma_4h_16[i] < hma_4h_48[i]) and (prev_hma_16[i] >= prev_hma_48[i])
            
            # Donchian breakout confirmation
            donch_breakout_up = close[i] > donch_upper[i] * 0.998
            donch_breakout_down = close[i] < donch_lower[i] * 1.002
            
            # Long: HMA cross up + bull trend + Donchian confirmation OR strong bull trend
            if bull_trend and hma_slope_bull:
                if hma_cross_up:
                    new_signal = POSITION_SIZE
                elif hma_4h_16[i] > hma_4h_48[i] and donch_breakout_up:
                    new_signal = POSITION_SIZE * 0.8
            
            # Short: HMA cross down + bear trend + Donchian confirmation OR strong bear trend
            if new_signal == 0.0:
                if bear_trend and hma_slope_bear:
                    if hma_cross_down:
                        new_signal = -POSITION_SIZE
                    elif hma_4h_16[i] < hma_4h_48[i] and donch_breakout_down:
                        new_signal = -POSITION_SIZE * 0.8
        
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
        
        # === EXIT CONDITIONS (regime flip or extreme) ===
        # Exit long on regime flip to strong bear or extreme overbought
        if in_position and position_side > 0:
            if bear_trend and hma_slope_bear and trend_regime:
                new_signal = 0.0
            elif rsi_14[i] > 80.0:  # Extreme overbought
                new_signal = 0.0
            elif close[i] > bb_upper[i] * 1.02 and range_regime:  # BB breakout in range
                new_signal = 0.0
        
        # Exit short on regime flip to strong bull or extreme oversold
        if in_position and position_side < 0:
            if bull_trend and hma_slope_bull and trend_regime:
                new_signal = 0.0
            elif rsi_14[i] < 20.0:  # Extreme oversold
                new_signal = 0.0
            elif close[i] < bb_lower[i] * 0.98 and range_regime:  # BB breakdown in range
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