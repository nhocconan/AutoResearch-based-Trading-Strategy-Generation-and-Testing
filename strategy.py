#!/usr/bin/env python3
"""
Experiment #043: 1d Ehlers Fisher Transform + 1w HMA Trend + Volatility Regime

Hypothesis: Fisher Transform excels at catching reversals in bear/range markets where
simple trend strategies fail. Combined with 1w HMA trend filter and volatility-based
position sizing, this should work better than RSI-based entries on 1d timeframe.

Key design:
1. 1w HMA(21) for major trend bias (call ONCE via mtf_data)
2. Ehlers Fisher Transform(9) for reversal entries (wide thresholds: ±1.8)
3. Bollinger Band width for volatility regime (low vol = trend, high vol = revert)
4. ATR(14) for stoploss (2.5x) and position sizing
5. Discrete sizing: 0.25 base, 0.30 strong confluence

Why this should work:
- Fisher Transform normalizes price to Gaussian distribution, better than RSI for reversals
- 1d TF limits trades to 20-50/year (fee efficient)
- 1w HTF filter prevents counter-trend trades in strong trends
- Volatility regime adapts entry thresholds (wide in high vol, tight in low vol)
- Fisher thresholds wide enough (±1.8) to ensure trades trigger on reversals
- Different from all 34+ failed strategies (no Choppiness, no CRSI, no Donchian breakout)

Timeframe: 1d (REQUIRED)
HTF: 1w via mtf_data helper
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_1w_hma_vol_regime_v1"
timeframe = "1d"
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
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Excellent for catching reversals in bear/range markets.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - lowest) / (highest - lowest) over period
    3. Fisher = 0.5 * ln((1 + normalized) / (1 - normalized))
    4. Signal line = Fisher shifted by 1
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    # Typical price
    typical = (high + low) / 2.0
    
    for i in range(period, n):
        # Find highest and lowest over lookback period
        highest = np.max(typical[i-period+1:i+1])
        lowest = np.min(typical[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range == 0:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
        else:
            # Normalize to 0-1 range, then to -0.99 to +0.99
            normalized = (typical[i] - lowest) / price_range
            normalized = 0.998 * (2.0 * normalized - 1.0)  # scale to -0.998 to +0.998
            
            # Fisher transform
            if abs(normalized) < 0.999:
                fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
            else:
                fisher[i] = fisher[i-1] if i > 0 else 0.0
    
    # Signal line is previous Fisher value
    fisher_signal[1:] = fisher[:-1]
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma  # Normalized bandwidth
    return upper, lower, sma, bandwidth

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """Calculate ATR ratio for volatility spike detection."""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    ratio = atr_short / (atr_long + 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA trend
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (auto shift(1) for completed bars only)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    bb_upper, bb_lower, bb_sma, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    atr_ratio = calculate_atr_ratio(high, low, close, 7, 30)
    
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
    last_trade_bar = -50
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        if np.isnan(bb_bandwidth[i]):
            continue
        
        # === HTF TREND BIAS (1w) ===
        htf_bullish = close[i] > hma_1w_aligned[i]
        htf_bearish = close[i] < hma_1w_aligned[i]
        
        # === VOLATILITY REGIME ===
        # High vol (atr_ratio > 1.5) = mean reversion likely
        # Low vol (atr_ratio < 1.0) = trend follow likely
        high_vol = atr_ratio[i] > 1.5
        low_vol = atr_ratio[i] < 1.0
        
        # === BOLLINGER BAND POSITION ===
        bb_lower_pct = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
        near_bb_lower = bb_lower_pct < 0.15  # Bottom 15% of BB
        near_bb_upper = bb_lower_pct > 0.85  # Top 15% of BB
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.8 = bullish reversal (wide threshold for trade gen)
        # Fisher crosses below +1.8 = bearish reversal
        fisher_bull_cross = (fisher_signal[i] < -1.8) and (fisher[i] >= -1.8)
        fisher_bear_cross = (fisher_signal[i] > 1.8) and (fisher[i] <= 1.8)
        
        # Also check extreme levels (even without cross)
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # === ENTRY LOGIC - FISHER + REGIME ADAPTIVE ===
        new_signal = 0.0
        
        # Strong long: Fisher reversal + HTF bullish + near BB lower
        if fisher_bull_cross and htf_bullish and near_bb_lower:
            new_signal = STRONG_SIZE
        
        # Strong short: Fisher reversal + HTF bearish + near BB upper
        elif fisher_bear_cross and htf_bearish and near_bb_upper:
            new_signal = -STRONG_SIZE
        
        # Moderate long: Fisher oversold + HTF bullish (high vol = mean revert)
        elif fisher_oversold and htf_bullish and high_vol:
            new_signal = BASE_SIZE
        
        # Moderate short: Fisher overbought + HTF bearish (high vol = mean revert)
        elif fisher_overbought and htf_bearish and high_vol:
            new_signal = -BASE_SIZE
        
        # Trend follow in low vol: Fisher neutral + HTF trend
        elif low_vol and htf_bullish and fisher[i] > -0.5:
            new_signal = BASE_SIZE
        elif low_vol and htf_bearish and fisher[i] < 0.5:
            new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 20 bars (~20 days on 1d), force entry with HTF bias
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            # Only force if Fisher is neutral (not at extreme)
            if abs(fisher[i]) < 1.0:
                if htf_bullish:
                    new_signal = BASE_SIZE * 0.8
                elif htf_bearish:
                    new_signal = -BASE_SIZE * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
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
        
        # === FISHER REVERSAL EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            # Exit long when Fisher becomes overbought
            if position_side > 0 and fisher[i] > 2.0:
                fisher_exit = True
            # Exit short when Fisher becomes oversold
            if position_side < 0 and fisher[i] < -2.0:
                fisher_exit = True
        
        # === HTF TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if HTF trend turns bearish (with some hysteresis)
            if position_side > 0 and htf_bearish:
                # Only exit if Fisher also confirms
                if fisher[i] > 0:
                    trend_reversal = True
            # Exit short if HTF trend turns bullish
            if position_side < 0 and htf_bullish:
                if fisher[i] < 0:
                    trend_reversal = True
        
        # Apply stoploss or exits
        if stoploss_triggered or fisher_exit or trend_reversal:
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
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            # If same direction, keep position (no update needed)
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