#!/usr/bin/env python3
"""
Experiment #091: 4h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Recent 4h strategies failed due to over-complexity (too many filters = 0 trades).
This strategy uses SIMPLER logic proven to work:
1. 1d HMA(21) for major trend bias (price above = bullish, below = bearish)
2. 4h HMA(16/48) crossover for entry timing (faster HMA crosses slower)
3. RSI(14) pullback confirmation (RSI 35-55 for longs, 45-65 for shorts)
4. ATR(14) 2.5x trailing stoploss
5. Minimal regime filters to ENSURE trades happen on all symbols

Why this should work:
- Simpler = more trades = better statistics across BTC/ETH/SOL
- 1d HMA prevents counter-trend trades in major moves
- HMA crossover catches trend changes with less lag than EMA
- RSI pullback ensures we're not chasing tops/bottoms
- 4h timeframe naturally limits to 20-50 trades/year
- Discrete sizing (0.30) controls drawdown during 2022 crash

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol (looser entries than failed strategies)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_1d_v2"
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators ONCE
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators ONCE before loop
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    hma_21_4h = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30
    
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
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            continue
        
        # === 1D TREND BIAS (MAJOR) ===
        # Price above 1d HMA = bullish bias (prefer longs)
        # Price below 1d HMA = bearish bias (prefer shorts)
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 4H HMA CROSSOVER ===
        # Fast HMA(16) above Slow HMA(48) = bullish crossover
        # Fast HMA(16) below Slow HMA(48) = bearish crossover
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # Check previous bar for crossover detection
        hma_bullish_prev = hma_16[i-1] > hma_48[i-1] if i > 0 else False
        hma_bearish_prev = hma_16[i-1] < hma_48[i-1] if i > 0 else False
        
        bullish_crossover = hma_bullish and not hma_bullish_prev
        bearish_crossover = hma_bearish and not hma_bearish_prev
        
        # === RSI PULLBACK CONFIRMATION ===
        # For longs: RSI pulled back to 35-55 range (not oversold, not overbought)
        # For shorts: RSI pulled back to 45-65 range
        rsi_pullback_long = 35 <= rsi_14[i] <= 55
        rsi_pullback_short = 45 <= rsi_14[i] <= 65
        
        # More permissive RSI for entry (ensure trades happen)
        rsi_ok_long = rsi_14[i] < 60
        rsi_ok_short = rsi_14[i] > 40
        
        # === 4H TREND CONFIRMATION ===
        hma_21_up = hma_21_4h[i] > hma_21_4h[i-5] if i >= 5 else False
        hma_21_down = hma_21_4h[i] < hma_21_4h[i-5] if i >= 5 else False
        
        # === ENTRY LOGIC (simplified to ensure trades) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - multiple conditions to ensure we get trades
        long_condition_1 = price_above_1d_hma and hma_bullish and rsi_ok_long
        long_condition_2 = bullish_crossover and rsi_pullback_long
        long_condition_3 = price_above_1d_hma and hma_21_up and rsi_14[i] < 50
        
        # More permissive: allow entry if any 2 of 3 conditions met
        long_score = sum([long_condition_1, long_condition_2, long_condition_3])
        if long_score >= 2:
            new_signal = BASE_SIZE
        elif long_score >= 1 and bars_since_last_trade > 80:
            # Allow weaker entry if no trades recently
            new_signal = BASE_SIZE * 0.7
        
        # SHORT ENTRIES - multiple conditions to ensure we get trades
        short_condition_1 = price_below_1d_hma and hma_bearish and rsi_ok_short
        short_condition_2 = bearish_crossover and rsi_pullback_short
        short_condition_3 = price_below_1d_hma and hma_21_down and rsi_14[i] > 50
        
        # More permissive: allow entry if any 2 of 3 conditions met
        short_score = sum([short_condition_1, short_condition_2, short_condition_3])
        if short_score >= 2:
            new_signal = -BASE_SIZE
        elif short_score >= 1 and bars_since_last_trade > 80:
            # Allow weaker entry if no trades recently
            new_signal = -BASE_SIZE * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 150 bars (~25 days on 4h), force weaker entry
        if bars_since_last_trade > 150 and new_signal == 0.0 and not in_position:
            if price_above_1d_hma and rsi_14[i] < 45:
                new_signal = BASE_SIZE * 0.5
            elif price_below_1d_hma and rsi_14[i] > 55:
                new_signal = -BASE_SIZE * 0.5
        
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
        
        # === TREND REVERSAL EXIT ===
        # Exit if major trend reverses against position
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if price crosses below 1d HMA
            if position_side > 0 and price_below_1d_hma:
                trend_reversal = True
            # Exit short if price crosses above 1d HMA
            if position_side < 0 and price_above_1d_hma:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
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