#!/usr/bin/env python3
"""
Experiment #007: 1d Primary + 1w HTF — Volatility Spike Mean Reversion

Hypothesis: Complex regime filters (Choppiness + Connors) have failed because they're
too restrictive for daily timeframes. Instead, use volatility spike detection which
naturally occurs 20-50 times/year during panic events.

Key insights from failed experiments:
- Connors RSI + Choppiness combinations generated too few trades or whipsawed
- Multi-regime logic is too complex for 1d timeframe
- BTC/ETH need asymmetric logic (bear market bias)

This strategy uses:
1. 1w HMA(21) for MAJOR trend bias (only short if price < 1w HMA)
2. ATR ratio (7/30) > 2.0 = volatility spike (panic/reversal opportunity)
3. Bollinger Band (20, 2.5) for entry timing (extreme = mean revert)
4. Asymmetric sizing: 0.30 for longs, 0.25 for shorts (bear bias)
5. ATR(14) trailing stop at 2.5x

Why this should work:
- Vol spikes occur 20-50 times/year naturally (meets trade frequency target)
- Mean reversion after panic has 70%+ win rate historically
- 1w trend filter prevents counter-trend trades in strong trends
- Wider BB (2.5 std) reduces false signals
- Asymmetric sizing accounts for crypto's long-term upward bias

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 20-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_volspike_bb_1w_hma_v1"
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

def calculate_bollinger_bands(close, period=20, std_dev=2.5):
    """Calculate Bollinger Bands with configurable std dev."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.5)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    LONG_SIZE = 0.30  # More aggressive on longs (crypto long bias)
    SHORT_SIZE = 0.25  # Conservative on shorts (bear market protection)
    
    # Track position state for stoploss
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
        
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === VOLATILITY SPIKE DETECTION ===
        # ATR(7) / ATR(30) > 2.0 = panic/extreme volatility
        atr_ratio = atr_7[i] / atr_30[i]
        vol_spike = atr_ratio > 2.0
        
        # === 1W TREND BIAS (MAJOR) ===
        # Price above 1w HMA = bullish major trend (prefer longs)
        # Price below 1w HMA = bearish major trend (prefer shorts)
        trend_1w_bullish = close[i] > hma_1w_21_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === BOLLINGER BAND POSITION ===
        # Price below lower BB = oversold (long opportunity)
        # Price above upper BB = overbought (short opportunity)
        price_below_bb = close[i] < bb_lower[i]
        price_above_bb = close[i] > bb_upper[i]
        
        # === POSITION SIZING ===
        current_long_size = LONG_SIZE
        current_short_size = SHORT_SIZE
        
        # Reduce size if no vol spike (normal conditions)
        if not vol_spike:
            current_long_size = LONG_SIZE * 0.6
            current_short_size = SHORT_SIZE * 0.6
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Require: Vol spike OR extreme BB breach + 1w trend not strongly bearish
        # Vol spike + BB lower = panic reversal opportunity
        if vol_spike and price_below_bb:
            new_signal = current_long_size
        elif price_below_bb and not trend_1w_bearish:
            # BB breach without vol spike, but 1w trend OK
            new_signal = current_long_size * 0.7
        
        # SHORT ENTRIES
        # Require: Vol spike OR extreme BB breach + 1w trend bearish
        # More conservative on shorts (only in bear trend or extreme vol)
        if vol_spike and price_above_bb and trend_1w_bearish:
            new_signal = -current_short_size
        elif price_above_bb and trend_1w_bearish and atr_ratio > 1.5:
            # BB breach in bear trend with elevated vol
            new_signal = -current_short_size * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~2 months on 1d), allow weaker entry
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if price_below_bb and atr_ratio > 1.3:
                new_signal = current_long_size * 0.5
            elif price_above_bb and trend_1w_bearish and atr_ratio > 1.3:
                new_signal = -current_short_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Long: trail highest price
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Short: trail lowest price
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_1w_bearish and close[i] > bb_mid[i]:
                # Long in newly bearish 1w trend, price recovered to mid BB
                trend_reversal = True
            if position_side < 0 and trend_1w_bullish and close[i] < bb_mid[i]:
                # Short in newly bullish 1w trend, price dropped to mid BB
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === TAKE PROFIT (partial) ===
        # Reduce position by half at 2R profit
        if in_position and position_side != 0 and new_signal != 0.0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * atr_14[i] and np.abs(new_signal) > 0.1:
                    new_signal = new_signal * 0.5  # Take half profit
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * atr_14[i] and np.abs(new_signal) > 0.1:
                    new_signal = new_signal * 0.5  # Take half profit
        
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
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.abs(new_signal) < np.abs(signals[i-1]) and signals[i-1] != 0:
                # Partial exit (take profit)
                pass  # Keep tracking, just reduced size
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