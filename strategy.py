#!/usr/bin/env python3
"""
Experiment #915: 1h Primary + 4h/1d HTF — Fisher Transform + HTF Trend + Vol Regime

Hypothesis: After 600+ failed strategies, the key insight is that BTC/ETH need 
MEAN REVERSION in bear/range markets (2022 crash, 2025 test period), not pure trend.

Key innovations:
1. Ehlers Fisher Transform (period=9) — catches reversals in bear rallies, proven edge
2. 4h HMA(21) for medium-term trend bias (direction filter)
3. 1d HMA(21) for macro regime (bull/bear market)
4. Volatility regime: ATR(7)/ATR(30) ratio detects vol spikes for better entries
5. RSI(14) as secondary confirmation (not primary signal)
6. Relaxed entry thresholds to guarantee 30-60 trades/year on 1h

Why this should work:
- Fisher Transform specifically designed for non-Gaussian price distributions
- Works in both trending AND ranging markets (unlike pure trend strategies)
- HTF filters prevent counter-trend trades that kill Sharpe
- Vol regime filter avoids entering during chaotic vol spikes
- Conservative sizing (0.25-0.30) limits drawdown during 2022-style crashes

Entry logic:
- LONG: Fisher < -1.5 (oversold) + 4h HMA bullish + 1d HMA neutral/bullish + vol_ratio < 2.0
- SHORT: Fisher > +1.5 (overbought) + 4h HMA bearish + 1d HMA neutral/bearish + vol_ratio < 2.0
- Relaxed: Fisher < -1.0 or > +1.0 with stronger HTF confluence

Exit logic:
- Fisher crosses opposite threshold (mean reversion complete)
- HTF trend reverses (4h HMA flips)
- ATR trailing stop (2.5x) for catastrophic moves

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 40-70 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_htf_trend_vol_regime_4h1d_atr_v1"
timeframe = "1h"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform — converts prices to Gaussian-like distribution.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - lowest) / (highest - lowest) * 1.99 + 0.01
    3. Fisher: 0.5 * ln((1 + normalized) / (1 - normalized))
    4. Signal line: 1-period lag of Fisher
    
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    Exit: Fisher crosses opposite threshold
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_signal
    
    for i in range(period - 1, n):
        # Get highest high and lowest low over lookback
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = 0.0
            continue
        
        # Typical price
        typical = (high[i] + low[i]) / 2.0
        
        # Normalize to range 0.01 to 1.99 (avoid division by zero)
        normalized = (typical - lowest) / (highest - lowest) * 1.99 + 0.01
        normalized = np.clip(normalized, 0.01, 1.99)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
        
        # Signal line (1-period lag)
        if i > period - 1:
            fisher_signal[i] = fisher[i - 1]
    
    return fisher, fisher_signal

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

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """ATR ratio for volatility regime detection."""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = atr_short / (atr_long + 1e-10)
    
    return ratio

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands for mean reversion reference."""
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    if n < period:
        return upper, lower, middle
    
    for i in range(period - 1, n):
        window = close[i-period+1:i+1]
        middle[i] = np.mean(window)
        std = np.std(window, ddof=0)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
    
    return upper, lower, middle

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    fisher_1h, fisher_signal_1h = calculate_fisher_transform(high, low, close, period=9)
    atr_1h = calculate_atr(high, low, close, period=14)
    atr_ratio_1h = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Calculate and align 4h HMA for medium-term trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro regime (bull/bear market)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
    
    # Track Fisher extreme for exit timing
    fisher_extreme_long = float('inf')
    fisher_extreme_short = float('-inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(fisher_1h[i]) or np.isnan(fisher_signal_1h[i]):
            continue
        if np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(atr_ratio_1h[i]):
            continue
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === VOLATILITY REGIME ===
        # High vol (ratio > 2.0) = avoid entries (chaotic)
        # Normal vol (ratio < 1.5) = good for entries
        low_vol_regime = atr_ratio_1h[i] < 1.8
        high_vol_regime = atr_ratio_1h[i] > 2.2
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher_1h[i] < -1.5
        fisher_overbought = fisher_1h[i] > 1.5
        fisher_moderate_oversold = fisher_1h[i] < -1.0
        fisher_moderate_overbought = fisher_1h[i] > 1.0
        
        # Fisher crossing (reversal signal)
        fisher_cross_long = (fisher_1h[i] > -1.5) and (fisher_signal_1h[i] <= -1.5) if not np.isnan(fisher_signal_1h[i]) else False
        fisher_cross_short = (fisher_1h[i] < 1.5) and (fisher_signal_1h[i] >= 1.5) if not np.isnan(fisher_signal_1h[i]) else False
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_1h[i] < 35
        rsi_overbought = rsi_1h[i] > 65
        rsi_extreme_oversold = rsi_1h[i] < 25
        rsi_extreme_overbought = rsi_1h[i] > 75
        
        # === BOLLINGER BAND POSITION ===
        near_bb_lower = close[i] < bb_lower[i] * 1.01  # Within 1% of lower band
        near_bb_upper = close[i] > bb_upper[i] * 0.99  # Within 1% of upper band
        
        desired_signal = 0.0
        
        # === PRIMARY ENTRY: Fisher extreme + HTF trend confluence ===
        # Long: Fisher oversold + 4h bullish OR 1d bullish + low vol
        if fisher_oversold and low_vol_regime:
            if trend_4h_bullish or macro_bull:
                # Strong confluence: both HTF agree
                if trend_4h_bullish and macro_bull:
                    desired_signal = BASE_SIZE
                else:
                    # One HTF agrees + RSI confirmation
                    if rsi_oversold:
                        desired_signal = BASE_SIZE
                    else:
                        desired_signal = REDUCED_SIZE
        
        # Short: Fisher overbought + 4h bearish OR 1d bearish + low vol
        if fisher_overbought and low_vol_regime:
            if trend_4h_bearish or macro_bear:
                # Strong confluence: both HTF agree
                if trend_4h_bearish and macro_bear:
                    desired_signal = -BASE_SIZE
                else:
                    # One HTF agrees + RSI confirmation
                    if rsi_overbought:
                        desired_signal = -BASE_SIZE
                    else:
                        desired_signal = -REDUCED_SIZE
        
        # === SECONDARY ENTRY: Moderate Fisher + BB + RSI confluence ===
        if desired_signal == 0.0 and low_vol_regime:
            # Long: Moderate Fisher oversold + near BB lower + RSI oversold
            if fisher_moderate_oversold and near_bb_lower and rsi_oversold:
                if trend_4h_bullish or macro_bull:
                    desired_signal = REDUCED_SIZE
            
            # Short: Moderate Fisher overbought + near BB upper + RSI overbought
            if fisher_moderate_overbought and near_bb_upper and rsi_overbought:
                if trend_4h_bearish or macro_bear:
                    desired_signal = -REDUCED_SIZE
        
        # === TERTIARY ENTRY: Fisher cross (reversal confirmation) ===
        if desired_signal == 0.0 and low_vol_regime:
            # Long on Fisher cross above -1.5
            if fisher_cross_long:
                if trend_4h_bullish or macro_bull:
                    desired_signal = REDUCED_SIZE
            
            # Short on Fisher cross below +1.5
            if fisher_cross_short:
                if trend_4h_bearish or macro_bear:
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
        
        # === EXIT CONDITIONS ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Exit long if Fisher becomes overbought (mean reversion complete)
                if fisher_1h[i] > 1.0:
                    desired_signal = 0.0
                # Exit if HTF trend reverses strongly
                elif trend_4h_bearish and macro_bear:
                    desired_signal = 0.0
                # Hold if trend intact and Fisher not overbought
                elif (trend_4h_bullish or macro_bull) and fisher_1h[i] < 1.5:
                    desired_signal = BASE_SIZE
            
            elif position_side < 0:
                # Exit short if Fisher becomes oversold (mean reversion complete)
                if fisher_1h[i] < -1.0:
                    desired_signal = 0.0
                # Exit if HTF trend reverses strongly
                elif trend_4h_bullish and macro_bull:
                    desired_signal = 0.0
                # Hold if trend intact and Fisher not oversold
                elif (trend_4h_bearish or macro_bear) and fisher_1h[i] > -1.5:
                    desired_signal = -BASE_SIZE
        
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
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
                fisher_extreme_long = fisher_1h[i] if position_side > 0 else float('inf')
                fisher_extreme_short = fisher_1h[i] if position_side < 0 else float('-inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
                fisher_extreme_long = fisher_1h[i] if position_side > 0 else float('inf')
                fisher_extreme_short = fisher_1h[i] if position_side < 0 else float('-inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
                fisher_extreme_long = min(fisher_extreme_long, fisher_1h[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
                fisher_extreme_short = max(fisher_extreme_short, fisher_1h[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
                fisher_extreme_long = float('inf')
                fisher_extreme_short = float('-inf')
        
        signals[i] = desired_signal
    
    return signals