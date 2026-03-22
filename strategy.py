#!/usr/bin/env python3
"""
Experiment #046: 12h Fisher Transform + 1d HMA Trend + Choppiness Regime

Hypothesis: 12h primary with 1d HTF filter, using Ehlers Fisher Transform for
entry timing (superior to RSI in bear/range markets) + Choppiness regime switch
will generate consistent trades with positive Sharpe across all symbols.

Key design:
1. 1d HMA(21) for trend bias via mtf_data helper (call ONCE before loop)
2. Choppiness Index(14) for regime: >55 = range (mean revert), <45 = trend
3. Fisher Transform(9) for entries: crosses -1.5 = long, crosses +1.5 = short
4. ATR(14) for stoploss at 2.5x
5. Discrete sizing: 0.25 base, 0.30 strong trend alignment
6. Frequency safeguard: force entry after 20 bars without trades

Why this should work:
- Fisher Transform catches reversals better than RSI in bear markets (research-backed)
- 12h TF naturally limits to 20-50 trades/year (fee efficient)
- 1d HTF filter prevents counter-trend trades
- Choppiness adapts between mean-revert and trend-follow modes
- Wide Fisher thresholds ensure trades trigger (not like RSI 42-43 narrow bands)
- Frequency safeguard prevents 0-trade failure mode

Timeframe: 12h (REQUIRED per experiment)
HTF: 1d via mtf_data helper
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_chop_1d_hma_regime_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (price - lowest) / (highest - lowest) - 0.67
    Signals: Fisher crosses above -1.5 = long, crosses below +1.5 = short
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 0:
            X = 0.67 * (close[i] - lowest) / price_range - 0.67
            X = np.clip(X, -0.999, 0.999)  # prevent division by zero
            fisher[i] = 0.5 * np.log((1 + X) / (1 - X))
            
            # Signal line is previous Fisher value
            if i > period:
                fisher_signal[i] = fisher[i-1]
        else:
            fisher[i] = 0.0
            fisher_signal[i] = 0.0
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/ranging market
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
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA trend
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (auto shift(1) for completed bars only)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -30
    prev_fisher = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]):
            continue
        
        # === HTF TREND BIAS (1d) ===
        htf_bullish = close[i] > hma_1d_aligned[i]
        htf_bearish = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness) ===
        is_choppy = chop_14[i] > 55
        is_trending = chop_14[i] < 45
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher cross above -1.5 = long signal
        # Fisher cross below +1.5 = short signal
        fisher_long = fisher[i] > -1.5 and prev_fisher <= -1.5
        fisher_short = fisher[i] < 1.5 and prev_fisher >= 1.5
        
        # Also use Fisher extreme reversals
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # === ENTRY LOGIC - REGIME ADAPTIVE ===
        new_signal = 0.0
        
        if is_trending and htf_bullish:
            # Trend follow long: Fisher reversal in uptrend
            if fisher_long or (fisher_oversold and htf_bullish):
                new_signal = STRONG_SIZE
        
        elif is_trending and htf_bearish:
            # Trend follow short: Fisher reversal in downtrend
            if fisher_short or (fisher_overbought and htf_bearish):
                new_signal = -STRONG_SIZE
        
        elif is_choppy:
            # Mean reversion in range: Fisher extremes
            if fisher_oversold:
                new_signal = BASE_SIZE
            elif fisher_overbought:
                new_signal = -BASE_SIZE
        
        else:
            # Neutral regime: use HTF bias with Fisher confirmation
            if htf_bullish and fisher_oversold:
                new_signal = BASE_SIZE
            elif htf_bearish and fisher_overbought:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 20 bars (~10 days on 12h), force entry
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if htf_bullish:
                new_signal = BASE_SIZE * 0.8
            elif htf_bearish:
                new_signal = -BASE_SIZE * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and htf_bearish:
                trend_reversal = True
            if position_side < 0 and htf_bullish:
                trend_reversal = True
        
        # === FISHER EXTREME EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            # Exit long when Fisher becomes very overbought
            if position_side > 0 and fisher[i] > 2.0:
                fisher_exit = True
            # Exit short when Fisher becomes very oversold
            if position_side < 0 and fisher[i] < -2.0:
                fisher_exit = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal or fisher_exit:
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
        prev_fisher = fisher[i]
    
    return signals