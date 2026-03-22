#!/usr/bin/env python3
"""
Experiment #039: 4h Donchian Breakout + 1d HMA Trend + Choppiness Regime

Hypothesis: 4h primary timeframe with 1d trend filter will generate 20-50 trades/year
with better risk-adjusted returns than pure trend-following. Choppiness Index switches
between breakout (trending) and mean-reversion (ranging) modes.

Key design:
1. 1d HMA(21) for major trend bias (call ONCE via mtf_data)
2. Choppiness Index(14) regime: >55 = range (mean revert), <45 = trend (breakout)
3. Donchian(20) breakout for entry timing in trending regime
4. RSI(14) filter for entry quality (avoid extreme entries)
5. ATR(14) trailing stop at 2.5x for risk management
6. Discrete sizing: 0.25 base, 0.30 strong trend confluence

Why this should work:
- 4h TF naturally limits trades to 30-60/year (fee efficient per Rule 10)
- 1d HTF filter prevents counter-trend breakouts
- Choppiness adapts entry logic to market regime
- Donchian breakouts generate reliable signals in trending markets
- RSI filter avoids chasing extended moves
- Wide RSI thresholds (25-65 long, 35-75 short) ensure trade generation

Timeframe: 4h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete (max 0.40 per Rule 4)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_chop_1d_hma_rsi_atr_v1"
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
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    loss_avg = np.where(loss_avg == 0, 1e-10, loss_avg)
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA trend
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (auto shift(1) for completed bars only - Rule 2)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        
        # === HTF TREND BIAS (1d) ===
        htf_bullish = close[i] > hma_1d_aligned[i]
        htf_bearish = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness) ===
        # CHOP > 55 = ranging (mean revert)
        # CHOP < 45 = trending (breakout)
        # 45 - 55 = neutral (use trend bias)
        is_choppy = chop_14[i] > 55
        is_trending = chop_14[i] < 45
        
        # === ENTRY LOGIC - REGIME ADAPTIVE ===
        new_signal = 0.0
        
        # Breakout detection
        breakout_long = close[i] > donch_upper[i-1]  # Break above previous upper
        breakout_short = close[i] < donch_lower[i-1]  # Break below previous lower
        
        if is_trending:
            # Trend-following breakout mode
            if htf_bullish and breakout_long:
                # RSI filter: not too overbought (wide range for trade gen)
                if 25 <= rsi_14[i] <= 65:
                    new_signal = STRONG_SIZE
            
            elif htf_bearish and breakout_short:
                # RSI filter: not too oversold (wide range for trade gen)
                if 35 <= rsi_14[i] <= 75:
                    new_signal = -STRONG_SIZE
        
        elif is_choppy:
            # Mean reversion mode in range
            # Buy at Donchian lower, sell at Donchian upper
            if close[i] <= donch_lower[i] * 1.002:  # Near lower band
                if rsi_14[i] < 50:
                    new_signal = BASE_SIZE
            
            elif close[i] >= donch_upper[i] * 0.998:  # Near upper band
                if rsi_14[i] > 50:
                    new_signal = -BASE_SIZE
        
        else:
            # Neutral regime: use HTF bias with breakout confirmation
            if htf_bullish and breakout_long:
                if rsi_14[i] < 60:
                    new_signal = BASE_SIZE
            
            elif htf_bearish and breakout_short:
                if rsi_14[i] > 40:
                    new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 30 bars (~5 days on 4h), force entry to ensure trade gen
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if htf_bullish and rsi_14[i] < 55:
                new_signal = BASE_SIZE * 0.8
            elif htf_bearish and rsi_14[i] > 45:
                new_signal = -BASE_SIZE * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if HTF trend turns bearish
            if position_side > 0 and htf_bearish:
                trend_reversal = True
            # Exit short if HTF trend turns bullish
            if position_side < 0 and htf_bullish:
                trend_reversal = True
        
        # === RSI EXTREME EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long when RSI becomes very overbought
            if position_side > 0 and rsi_14[i] > 80:
                rsi_exit = True
            # Exit short when RSI becomes very oversold
            if position_side < 0 and rsi_14[i] < 20:
                rsi_exit = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal or rsi_exit:
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