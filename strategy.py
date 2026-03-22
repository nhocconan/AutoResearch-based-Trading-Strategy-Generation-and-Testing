#!/usr/bin/env python3
"""
Experiment #029: 4h HMA-Donchian Breakout with 1d Trend Filter

Hypothesis: Building on #026's 12h success (Sharpe=-0.319, best recent), adapt to 4h:
1. 4h HMA(16/48) crossover for trend direction (proven in #026)
2. 4h Donchian(20) breakout for entry timing (proven in #026)
3. 1d HMA(21) for major trend bias (bonus, not hard requirement)
4. RSI(14) filter to avoid extreme entries
5. 2.5 ATR trailing stoploss

Key improvements from #026:
- Looser entry scoring (>=1.5 instead of >=2.0) to ensure trade frequency
- 1d filter is bonus confirmation only (not required) to avoid 0 trades
- Fixed stoploss initialization for short positions
- Added minimum bars-in-position before exit to avoid premature exits

Timeframe: 4h (REQUIRED per experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target: 30-60 trades/year (4h = ~2190 bars/year, need ~1-2 trades/week)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_donchian_1d_bias_rsi_atr_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_4h_16 = calculate_hma(close, 16)
    hma_4h_48 = calculate_hma(close, 48)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 999999.0  # Initialize high for short stoploss
    bars_since_entry = 0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1D TREND BIAS (bonus, not required) ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H HMA TREND ===
        hma_bullish = hma_4h_16[i] > hma_4h_48[i]
        hma_bearish = hma_4h_16[i] < hma_4h_48[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI FILTER (avoid extreme overbought/oversold entries) ===
        rsi_ok_long = rsi_14[i] < 75
        rsi_ok_short = rsi_14[i] > 25
        
        # === SCORING SYSTEM (looser than #026 to ensure trades) ===
        # LONG: Need trend (required) + 0.5 points from breakout/daily/RSI
        long_score = 0.0
        if hma_bullish:
            long_score += 1.0  # Required
        if breakout_long:
            long_score += 0.7
        if daily_bullish:
            long_score += 0.4
        if rsi_ok_long:
            long_score += 0.4
        
        # SHORT: Need trend (required) + 0.5 points from breakout/daily/RSI
        short_score = 0.0
        if hma_bearish:
            short_score += 1.0  # Required
        if breakout_short:
            short_score += 0.7
        if daily_bearish:
            short_score += 0.4
        if rsi_ok_short:
            short_score += 0.4
        
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Need score >= 1.5 (trend + at least one other signal)
        if long_score >= 1.5 and hma_bullish:
            new_signal = BASE_SIZE
        
        # SHORT ENTRY: Need score >= 1.5
        if short_score >= 1.5 and hma_bearish:
            new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~13 days on 4h), allow weaker entry
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if hma_bullish and rsi_14[i] < 60:
                new_signal = BASE_SIZE * 0.7
            elif hma_bearish and rsi_14[i] > 40:
                new_signal = -BASE_SIZE * 0.7
        
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
                if close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT (only after min bars in position) ===
        trend_reversal = False
        if in_position and position_side != 0 and bars_since_entry > 5:
            # Exit long if 4h HMA turns bearish
            if position_side > 0 and hma_bearish:
                trend_reversal = True
            # Exit short if 4h HMA turns bullish
            if position_side < 0 and hma_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 999999.0
                lowest_price = close[i] if position_side < 0 else 999999.0
                bars_since_entry = 0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 999999.0
                lowest_price = close[i] if position_side < 0 else 999999.0
                bars_since_entry = 0
                last_trade_bar = i
            else:
                # Same direction, maintain position
                bars_since_entry += 1
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 999999.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals