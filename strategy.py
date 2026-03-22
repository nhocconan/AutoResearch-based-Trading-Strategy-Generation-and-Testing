#!/usr/bin/env python3
"""
Experiment #024: 4h Volatility Spike Mean Reversion + HTF Trend Filter

Hypothesis: After 23 failed experiments with regime-switching and complex multi-filter
strategies, return to a simpler volatility-based mean reversion approach that has
proven success on BTC/ETH during crashes.

Key insights from failures:
- Regime-switching (Choppiness) consistently failed (#015, #019, #022, #023)
- RSI pullback strategies generated 0 trades or negative Sharpe (#014, #017, #018, #020)
- 12h primary timeframe failed (#012, #022) - too slow, late entries
- Too many HTF filters = 0 trades or late entries

This strategy uses:
1. Volatility spike detection: ATR(7)/ATR(30) > 2.0 (panic/capitulation)
2. Mean reversion entry: Price outside BB(20, 2.5) = extreme deviation
3. HTF trend filter: 12h HMA(21) for direction, 1d HMA(21) for regime bias
4. ATR(14) trailing stop: 2.5 ATR exit to protect capital
5. Volatility crush exit: ATR ratio < 1.2 = volatility normalized

Why this should work:
- Volatility spikes ALWAYS reverse (mean reversion edge)
- HTF filter prevents counter-trend entries in strong trends
- 4h timeframe = 30-60 trades/year naturally (not too many fees)
- Simple logic = fewer conflicting filters = more trades generated
- Discrete sizing (0.25/0.30) = minimizes fee churn

Timeframe: 4h (REQUIRED for this experiment)
HTF: 12h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_spike_meanrev_12h_1d_hma_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_dev=2.5):
    """Calculate Bollinger Bands with configurable std dev."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper.values, lower.values, sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.5)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === VOLATILITY SPIKE DETECTION ===
        atr_ratio = atr_7[i] / atr_30[i]
        vol_spike = atr_ratio > 2.0
        vol_normalized = atr_ratio < 1.2
        
        # === MEAN REVERSION SIGNALS ===
        price_below_bb = close[i] < bb_lower[i]
        price_above_bb = close[i] > bb_upper[i]
        
        # === HTF TREND FILTER ===
        # 12h HMA for immediate trend direction
        trend_12h_bullish = close[i] > hma_12h_21_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_21_aligned[i]
        
        # 1d HMA for regime bias
        regime_1d_bullish = close[i] > hma_1d_21_aligned[i]
        regime_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: Vol spike + price below BB + HTF trend not strongly bearish
        # Allow long in bear market if vol spike is extreme (capitulation)
        long_entry = (
            vol_spike and
            price_below_bb and
            (trend_12h_bullish or (regime_1d_bearish and atr_ratio > 2.5))
        )
        
        # SHORT ENTRY: Vol spike + price above BB + HTF trend not strongly bullish
        short_entry = (
            vol_spike and
            price_above_bb and
            (trend_12h_bearish or (regime_1d_bullish and atr_ratio > 2.5))
        )
        
        if long_entry:
            new_signal = BASE_SIZE
        
        if short_entry:
            new_signal = -BASE_SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === VOLATILITY CRUSH EXIT ===
        # Exit when volatility normalizes (mean reversion complete)
        vol_exit = False
        if in_position and vol_normalized:
            vol_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 12h trend turns strongly bearish
            if position_side > 0 and trend_12h_bearish and regime_1d_bearish:
                trend_reversal = True
            # Exit short if 12h trend turns strongly bullish
            if position_side < 0 and trend_12h_bullish and regime_1d_bullish:
                trend_reversal = True
        
        # Apply stoploss, vol exit, or trend reversal
        if stoploss_triggered or vol_exit or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals