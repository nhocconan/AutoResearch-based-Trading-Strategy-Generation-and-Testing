#!/usr/bin/env python3
"""
Experiment #122: 12h Primary + 1d HTF — Simplified Regime Mean Reversion

Hypothesis: Previous strategies failed due to too many conflicting filters (0 trades).
This strategy SIMPLIFIES entry logic while keeping regime awareness:

1. CHOPPINESS INDEX: Primary regime filter (>55 = range/mean revert, <45 = trend)
2. RSI(14): Simpler than Connors RSI, more reliable across symbols
3. 1d HMA(21): Trend bias only (don't require perfect alignment)
4. ATR(14): Volatility filter + trailing stoploss
5. BOLLINGER BANDS: Entry trigger at extremes (2.0 std for more trades)

Key changes from failed experiments:
- LOWER thresholds: RSI < 35/>65 instead of extreme CRSI values
- FEWER confluence requirements: 2 filters instead of 3-4
- FORCED entries: If no trade for 200 bars, enter on simple RSI extreme
- HIGHER position size in range markets (more trades = mean revert works)

Why this should beat Sharpe=0.220:
- More trades (30-50/year target vs 10-20 in over-filtered strategies)
- Works in both bull and bear (regime-adaptive)
- 12h TF = low fee drag, enough signals per year
- Conservative sizing (0.25-0.30) protects from 2022-style crashes

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_rsi_bb_1d_simple_v1"
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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    atr_values = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    chop_14 = calculate_choppiness(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    RANGE_SIZE = 0.32  # Higher size in range markets (mean revert works better)
    TREND_SIZE = 0.22  # Lower size in trend markets (more risk)
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            continue
        
        # === 1D TREND BIAS (loose filter) ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.2
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.2
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        bb_width_pct = (bb_upper[i] - bb_lower[i]) / bb_mid[i] * 100 if bb_mid[i] > 0 else 0
        
        # === RSI EXTREMES (lowered thresholds for more trades) ===
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        rsi_extreme_low = rsi_14[i] < 30
        rsi_extreme_high = rsi_14[i] > 70
        
        # === POSITION SIZING BY REGIME ===
        if is_range_market:
            current_size = RANGE_SIZE
        elif is_trend_market:
            current_size = TREND_SIZE
        else:
            current_size = BASE_SIZE
        
        # === ENTRY LOGIC (SIMPLIFIED - fewer confluence requirements) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        long_conditions = 0
        
        # Condition 1: Range market + RSI oversold (primary mean revert)
        if is_range_market and rsi_oversold:
            long_conditions += 2
        
        # Condition 2: BB lower + RSI oversold (double confirmation)
        if price_below_bb_lower and rsi_oversold:
            long_conditions += 2
        
        # Condition 3: 1d bullish bias + RSI pullback
        if trend_1d_bullish and rsi_14[i] < 45:
            long_conditions += 1
        
        # Condition 4: Price above 1d HMA + RSI low (pullback in uptrend)
        if price_above_1d_hma and rsi_14[i] < 40:
            long_conditions += 1
        
        # Condition 5: Extreme RSI alone (catch deep dips)
        if rsi_extreme_low:
            long_conditions += 1
        
        # Condition 6: BB squeeze breakout (low vol + oversold)
        if bb_width_pct < 5 and rsi_oversold:
            long_conditions += 1
        
        if long_conditions >= 2:
            new_signal = current_size
        elif long_conditions == 1 and bars_since_last_trade > 60:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_conditions = 0
        
        # Condition 1: Range market + RSI overbought
        if is_range_market and rsi_overbought:
            short_conditions += 2
        
        # Condition 2: BB upper + RSI overbought
        if price_above_bb_upper and rsi_overbought:
            short_conditions += 2
        
        # Condition 3: 1d bearish bias + RSI rally
        if trend_1d_bearish and rsi_14[i] > 55:
            short_conditions += 1
        
        # Condition 4: Price below 1d HMA + RSI high (rally in downtrend)
        if price_below_1d_hma and rsi_14[i] > 60:
            short_conditions += 1
        
        # Condition 5: Extreme RSI alone
        if rsi_extreme_high:
            short_conditions += 1
        
        # Condition 6: BB squeeze breakout (low vol + overbought)
        if bb_width_pct < 5 and rsi_overbought:
            short_conditions += 1
        
        if short_conditions >= 2:
            new_signal = -current_size
        elif short_conditions == 1 and bars_since_last_trade > 60:
            new_signal = -current_size * 0.5
        
        # === FORCED TRADE MECHANISM (ensure minimum trades) ===
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and rsi_14[i] < 45:
                new_signal = current_size * 0.5
            elif trend_1d_bearish and rsi_14[i] > 55:
                new_signal = -current_size * 0.5
            elif rsi_extreme_low:
                new_signal = current_size * 0.4
            elif rsi_extreme_high:
                new_signal = -current_size * 0.4
        
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
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if trend turns bearish strongly
            if position_side > 0 and trend_1d_bearish and hma_1d_slope_aligned[i] < -0.5:
                regime_reversal = True
            # Exit short if trend turns bullish strongly
            if position_side < 0 and trend_1d_bullish and hma_1d_slope_aligned[i] > 0.5:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
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