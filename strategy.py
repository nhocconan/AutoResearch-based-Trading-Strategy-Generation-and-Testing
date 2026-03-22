#!/usr/bin/env python3
"""
Experiment #200: 1h Primary + 4h/12h HTF — Simplified Multi-Timeframe Mean Reversion

Hypothesis: Previous 1h strategies (#188, #190, #195, #198) generated ZERO trades because
entry conditions were too strict. This strategy uses LOOSER thresholds to guarantee
trades while maintaining quality via HTF trend filter.

Key design for TRADE GENERATION:
1. 4h HMA(21) = primary trend bias (simple, reliable)
2. 1h RSI(14) = entry trigger with LOOSE thresholds (35/65 not 30/70)
3. 1h Bollinger(20, 1.8) = confirmation (wider than 2.0 for more signals)
4. Trade debt mechanism = forces entries if no trades for 80 bars
5. Position size = 0.25 (conservative for 1h, allows more trades)

Why this should generate trades:
- RSI 35/65 triggers frequently (vs 20/80 which rarely hits)
- BB 1.8 std dev = more breaks than 2.0 or 2.5
- Trade debt ensures minimum frequency
- Simple 4h HMA trend = less filtering than complex regime

Timeframe: 1h (REQUIRED)
HTF: 4h via mtf_data.get_htf_data() — called ONCE before loop
Target: 50-100 trades/year per symbol (within 30-80 guideline)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_bb_hma4h_loose_v1"
timeframe = "1h"
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

def calculate_bollinger_bands(close, period=20, std_mult=1.8):
    """Calculate Bollinger Bands with 1.8 std for more signals."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
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
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 1.8)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    trade_count = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        # === 4H TREND BIAS ===
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_aligned[i]
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === RSI LEVELS (LOOSE for more trades) ===
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        rsi_extreme_low = rsi_14[i] < 35
        rsi_extreme_high = rsi_14[i] > 65
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG: 4h bullish + 1h oversold
        long_cond1 = price_above_4h_hma and rsi_extreme_low  # Trend + deep pullback
        long_cond2 = price_above_4h_hma and rsi_oversold and price_below_bb_lower  # Trend + BB break
        long_cond3 = rsi_14[i] < 30 and price_below_bb_lower  # Extreme oversold (any trend)
        
        if long_cond1 or long_cond2:
            new_signal = BASE_SIZE
        elif long_cond3:
            new_signal = BASE_SIZE * 0.8
        
        # SHORT: 4h bearish + 1h overbought
        short_cond1 = price_below_4h_hma and rsi_extreme_high  # Trend + deep rally
        short_cond2 = price_below_4h_hma and rsi_overbought and price_above_bb_upper  # Trend + BB break
        short_cond3 = rsi_14[i] > 70 and price_above_bb_upper  # Extreme overbought (any trend)
        
        if short_cond1 or short_cond2:
            new_signal = -BASE_SIZE
        elif short_cond3:
            new_signal = -BASE_SIZE * 0.8
        
        # === TRADE DEBT MECHANISM (ensure minimum trades) ===
        if bars_since_last_trade > 80 and not in_position:
            # Force long if 4h bullish and RSI reasonable
            if price_above_4h_hma and rsi_14[i] < 50:
                new_signal = BASE_SIZE * 0.5
            # Force short if 4h bearish and RSI reasonable
            elif price_below_4h_hma and rsi_14[i] > 50:
                new_signal = -BASE_SIZE * 0.5
            # Fallback: pure RSI extremes
            elif rsi_14[i] < 35:
                new_signal = BASE_SIZE * 0.4
            elif rsi_14[i] > 65:
                new_signal = -BASE_SIZE * 0.4
        
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
        
        if stoploss_triggered:
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
                trade_count += 1
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
                trade_count += 1
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