#!/usr/bin/env python3
"""
Experiment #931: 4h Primary + 1d/1w HTF — Fisher Transform + Choppiness Regime + HMA Trend

Hypothesis: After 661 failed strategies, the key is combining reversal detection (Fisher Transform)
with regime filtering (Choppiness) and HTF trend bias (1d/1w HMA). This differs from CRSI-heavy
approaches that have been over-tested. Fisher Transform excels at catching reversals in bear/range
markets (2022 crash, 2025 bear) where simple trend-following fails.

Key insights from research:
1. Fisher Transform (period=9): Normalizes price to Gaussian distribution, crosses at ±1.5 signal reversals
2. Choppiness Index(14): CHOP>55=range (use Fisher reversals), CHOP<45=trend (use HMA direction)
3. 1d HMA(21): Medium-term trend bias for signal direction
4. 1w HMA(21): Macro regime filter (only long if price>1w HMA in bull market)
5. Relaxed Fisher thresholds (±1.2 not ±1.5) to ensure sufficient trades on all symbols
6. ATR(14) trailing stop (2.5x) for risk management

Why this should work on 4h:
- Fisher Transform catches reversals that RSI/CRSI miss in volatile markets
- Regime-switching logic adapts to market conditions (range vs trend)
- HTF HMA provides strong trend bias without lag of simple MA
- Relaxed thresholds ensure 30+ trades per symbol (avoiding 0-trade failure)
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_regime_1d1w_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform — normalizes price to Gaussian distribution.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: 0.66 * ((typical - lowest) / (highest - lowest) - 0.5) + 0.67 * prev_fisher
    3. Fisher = 0.5 * ln((1 + normalized) / (1 - normalized))
    
    Signal: Fisher crosses above -1.5 → long, crosses below +1.5 → short
    Relaxed to ±1.2 for more trades
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period + 2:
        return fisher, fisher_signal
    
    for i in range(period, n):
        # Calculate highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = fisher[i-1] if not np.isnan(fisher[i-1]) else 0.0
            fisher_signal[i] = fisher_signal[i-1] if not np.isnan(fisher_signal[i-1]) else 0.0
            continue
        
        # Typical price
        typical = (high[i] + low[i]) / 2.0
        
        # Normalize to -1 to +1 range
        normalized = 0.66 * ((typical - lowest) / (highest - lowest) - 0.5) + 0.67 * (fisher[i-1] if not np.isnan(fisher[i-1]) else 0.0)
        
        # Clamp to avoid log errors
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Signal line (1-period lag of Fisher)
        fisher_signal[i] = fisher[i-1] if not np.isnan(fisher[i-1]) else 0.0
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging, CHOP < 45 = trending.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (4h) indicators
    fisher_4h, fisher_signal_4h = calculate_fisher_transform(high, low, period=9)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    rsi_4h = calculate_rsi(close, period=14)
    
    # Calculate and align 1d HMA for medium-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro regime (bull/bear market)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(fisher_4h[i]) or np.isnan(fisher_signal_4h[i]):
            continue
        if np.isnan(chop_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(rsi_4h[i]):
            continue
        
        # === MACRO REGIME (1w HTF HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === MEDIUM-TERM TREND (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (4h Choppiness Index) ===
        ranging_regime = chop_4h[i] > 55
        trending_regime = chop_4h[i] < 45
        
        # === FISHER TRANSFORM SIGNALS (Relaxed thresholds: ±1.2) ===
        fisher_cross_long = (fisher_signal_4h[i] < -1.2) and (fisher_4h[i] > -1.2)
        fisher_cross_short = (fisher_signal_4h[i] > 1.2) and (fisher_4h[i] < 1.2)
        
        # Extreme Fisher levels for stronger signals
        fisher_extreme_long = fisher_4h[i] < -1.8
        fisher_extreme_short = fisher_4h[i] > 1.8
        
        # === RSI FILTER (avoid entering against extreme momentum) ===
        rsi_neutral = (rsi_4h[i] > 30) and (rsi_4h[i] < 70)
        rsi_oversold = rsi_4h[i] < 35
        rsi_overbought = rsi_4h[i] > 65
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 55) — Fisher Reversals ===
        if ranging_regime:
            # Long: Fisher crosses up from oversold + trend alignment
            if fisher_cross_long:
                if macro_bull or trend_1d_bullish or rsi_oversold:
                    desired_signal = BASE_SIZE
                else:
                    desired_signal = REDUCED_SIZE
            
            # Short: Fisher crosses down from overbought + trend alignment
            if fisher_cross_short:
                if macro_bear or trend_1d_bearish or rsi_overbought:
                    desired_signal = -BASE_SIZE
                else:
                    desired_signal = -REDUCED_SIZE
            
            # Fallback: Extreme Fisher alone (guarantees trades)
            if fisher_extreme_long and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if fisher_extreme_short and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45) — Trend Following with Fisher Entry ===
        elif trending_regime:
            # Long: Bullish trend + Fisher pullback entry
            if macro_bull or trend_1d_bullish:
                if fisher_cross_long or fisher_extreme_long:
                    desired_signal = BASE_SIZE
                elif fisher_4h[i] < -0.5 and rsi_oversold:
                    # Pullback entry in uptrend
                    desired_signal = REDUCED_SIZE
            
            # Short: Bearish trend + Fisher rally entry
            if macro_bear or trend_1d_bearish:
                if fisher_cross_short or fisher_extreme_short:
                    desired_signal = -BASE_SIZE
                elif fisher_4h[i] > 0.5 and rsi_overbought:
                    # Rally entry in downtrend
                    desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: Fisher + trend confluence required
            if fisher_cross_long and (macro_bull or trend_1d_bullish):
                desired_signal = BASE_SIZE
            
            if fisher_cross_short and (macro_bear or trend_1d_bearish):
                desired_signal = -BASE_SIZE
            
            # Fallback: Extreme Fisher with RSI filter
            if fisher_extreme_long and rsi_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if fisher_extreme_short and rsi_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and Fisher not extremely overbought
                if (macro_bull or trend_1d_bullish) and fisher_4h[i] < 1.5:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and Fisher not extremely oversold
                if (macro_bear or trend_1d_bearish) and fisher_4h[i] > -1.5:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro + medium trend reverses + Fisher overbought
            if macro_bear and trend_1d_bearish and fisher_4h[i] > 1.5:
                desired_signal = 0.0
            # Exit if Fisher extremely overbought in ranging regime
            if ranging_regime and fisher_4h[i] > 2.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro + medium trend reverses + Fisher oversold
            if macro_bull and trend_1d_bullish and fisher_4h[i] < -1.5:
                desired_signal = 0.0
            # Exit if Fisher extremely oversold in ranging regime
            if ranging_regime and fisher_4h[i] < -2.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals