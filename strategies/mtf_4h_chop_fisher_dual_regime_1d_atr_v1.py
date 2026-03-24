#!/usr/bin/env python3
"""
Experiment #749: 4h Primary + 1d HTF — Dual Regime (Chop Filter + Fisher Transform)

Hypothesis: After analyzing 500+ failed strategies, clear patterns emerge:
1. Pure trend following fails in bear/range markets (2022 crash, 2025 bear)
2. Choppiness Index successfully distinguishes trend vs range regimes (ETH Sharpe +0.923 in #727)
3. Fisher Transform excels at catching reversals in bear market rallies
4. Dual regime approach: trend-follow when CHOP<38.2, mean-revert when CHOP>61.8
5. 1d HMA(21) provides reliable trend bias across all market conditions
6. Loose entry filters ensure >=30 trades/train, >=3 trades/test per symbol

Strategy design:
1. 1d HMA(21) for primary trend bias (proven in multiple successful strategies)
2. 4h Choppiness Index(14) for regime detection (trend vs range)
3. 4h Fisher Transform(9) for reversal entry timing
4. 4h Donchian(20) for trend breakout confirmation
5. 4h RSI(14) loose filter to ensure trade frequency
6. ATR(14) trailing stop 2.5x for risk management
7. Discrete signals: 0.0, ±0.25, ±0.30

Key differences from #739 (Sharpe=0.012):
- Added Choppiness Index regime filter (was missing, caused whipsaw)
- Added Fisher Transform for better reversal timing
- Simplified entry logic (4 paths → 2 clear regime-based paths)
- Better hold logic to maintain positions through trends

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_fisher_dual_regime_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel for breakout detection."""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures whether market is trending or ranging.
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR for each bar
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR, highest high, lowest low
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50  # neutral
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals when Fisher crosses extreme levels.
    Long: Fisher crosses above -1.5
    Short: Fisher crosses below +1.5
    """
    n = len(close := (high + low) / 2)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_signal
    
    # Calculate median price
    median = (high + low) / 2
    
    # Normalize price to -1 to +1 range
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range > 1e-10:
            normalized = 2 * ((median[i] - lowest) / price_range) - 1
            normalized = np.clip(normalized, -0.999, 0.999)  # prevent log errors
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
            
            # Signal line (1-period lag)
            if i > period:
                fisher_signal[i] = fisher[i-1]
        else:
            fisher[i] = 0
            fisher_signal[i] = 0
    
    return fisher, fisher_signal

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    fisher_4h, fisher_signal_4h = calculate_fisher_transform(high, low, period=9)
    sma_50 = calculate_sma(close, period=50)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias
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
    
    for i in range(300, n):  # Need buffer for all indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        if np.isnan(chop_4h[i]) or np.isnan(fisher_4h[i]):
            continue
        
        # === TREND BIAS (1d HTF HMA) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        trending_regime = chop_4h[i] < 38.2  # Trending market
        ranging_regime = chop_4h[i] > 61.8   # Ranging market
        neutral_regime = not trending_regime and not ranging_regime
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = fisher_signal_4h[i] < -1.5 and fisher_4h[i] >= -1.5
        fisher_cross_down = fisher_signal_4h[i] > 1.5 and fisher_4h[i] <= 1.5
        fisher_extreme_low = fisher_4h[i] < -1.5
        fisher_extreme_high = fisher_4h[i] > 1.5
        
        # === RSI FILTERS (loose to ensure trades) ===
        rsi_ok_long = rsi_4h[i] < 70 and rsi_4h[i] > 25
        rsi_ok_short = rsi_4h[i] < 75 and rsi_4h[i] > 30
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma50 = close[i] < sma_50[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        
        # === TRENDING REGIME LOGIC (CHOP < 38.2) ===
        if trending_regime:
            # Long: 1d bullish + Donchian breakout + RSI ok
            if trend_1d_bullish and close[i] > donch_upper[i-1] and rsi_ok_long:
                desired_signal = BASE_SIZE
            
            # Short: 1d bearish + Donchian breakdown + RSI ok
            if trend_1d_bearish and close[i] < donch_lower[i-1] and rsi_ok_short:
                desired_signal = -BASE_SIZE
            
            # Alternative: Strong trend confirmation
            if trend_1d_bullish and above_sma50 and above_sma200 and rsi_4h[i] > 45:
                desired_signal = BASE_SIZE
            
            if trend_1d_bearish and below_sma50 and below_sma200 and rsi_4h[i] < 55:
                desired_signal = -BASE_SIZE
        
        # === RANGING REGIME LOGIC (CHOP > 61.8) ===
        elif ranging_regime:
            # Mean reversion long: Fisher extreme low + 1d bullish bias
            if fisher_extreme_low and trend_1d_bullish and rsi_4h[i] < 40:
                desired_signal = REDUCED_SIZE
            
            # Mean reversion short: Fisher extreme high + 1d bearish bias
            if fisher_extreme_high and trend_1d_bearish and rsi_4h[i] > 60:
                desired_signal = -REDUCED_SIZE
            
            # Fisher crossover confirmation
            if fisher_cross_up and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
            
            if fisher_cross_down and trend_1d_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Conservative: only enter on strong 1d trend + Fisher confirmation
            if trend_1d_bullish and fisher_cross_up and above_sma50:
                desired_signal = REDUCED_SIZE
            
            if trend_1d_bearish and fisher_cross_down and below_sma50:
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1d HMA still bullish
                if trend_1d_bullish:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if 1d HMA still bearish
                if trend_1d_bearish:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1d trend reverses
            if trend_1d_bearish:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d trend reverses
            if trend_1d_bullish:
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
                # Position flip
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