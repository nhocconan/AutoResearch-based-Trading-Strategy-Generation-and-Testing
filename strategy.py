#!/usr/bin/env python3
"""
Experiment #629: 4h Primary + 1d HTF — Choppiness Regime + HMA Trend + RSI Entry

Hypothesis: Recent 4h strategies failed due to too many filters (0 trades on #619-621, #628).
This strategy simplifies entry logic while adding Choppiness Index regime detection.

Key insights from 556 failed strategies:
1. 1d HTF works better than 1w for 4h primary (1w too slow = 0 trades)
2. Choppiness Index regime switch worked on ETH (Sharpe +0.923 in research)
3. HMA trend filter is faster than EMA for detecting direction changes
4. RSI extremes (not pullback zones) generate more trades
5. Simpler entry conditions = more trades (critical for meeting min trade requirement)

Regime Logic:
- CHOP > 61.8: Range market → Mean reversion (RSI < 30 long, RSI > 70 short)
- CHOP < 38.2: Trend market → Trend follow (HMA direction + RSI pullback 40-60)
- CHOP 38.2-61.8: Transition → No trades (avoid whipsaw)

Why this might beat Sharpe=0.520:
- Regime-adaptive: different logic for chop vs trend (proven on ETH)
- 1d HMA slope for major trend (slower than 12h, more reliable)
- RSI extremes in chop = high win rate mean reversion
- RSI pullback in trend = enter on dips/rallies with momentum
- Fewer filters than #624 = more trades (target 25-40/year)

Position sizing: 0.28 discrete (per Rule 4, max 0.40)
Target: 25-40 trades/year on 4h (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_hma_rsi_1d_v1"
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
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag.
    """
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8: Choppy/Range market (mean reversion)
    CHOP < 38.2: Trending market (trend following)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    
    # Avoid division by zero
    price_range = np.where(price_range < 1e-10, 1e-10, price_range)
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for primary trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    hma_4h = calculate_hma(close, period=21)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D TREND BIAS (HMA slope over 2 bars) ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-2] if i >= 2 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-2] if i >= 2 else False
        
        # Price relative to 1d HMA
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4H HMA SLOPE (2 bars) ===
        hma_4h_slope_bull = hma_4h[i] > hma_4h[i-2] if i >= 2 else False
        hma_4h_slope_bear = hma_4h[i] < hma_4h[i-2] if i >= 2 else False
        
        # === CHOPPINESS REGIME ===
        chop_range = chop_14[i] > 61.8  # Range/Chop market
        chop_trend = chop_14[i] < 38.2  # Trending market
        # chop 38.2-61.8 = transition, no trades
        
        # === RSI LEVELS ===
        rsi_oversold = rsi_14[i] < 30.0
        rsi_overbought = rsi_14[i] > 70.0
        rsi_pullback_long = 40.0 <= rsi_14[i] <= 55.0
        rsi_pullback_short = 45.0 <= rsi_14[i] <= 60.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- CHOPPY REGIME: Mean Reversion ---
        # Long when RSI oversold, Short when RSI overbought
        if chop_range:
            if rsi_oversold and hma_4h_slope_bull:
                new_signal = POSITION_SIZE
            elif rsi_overbought and hma_4h_slope_bear:
                new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Trend Following ---
        # Long: 1d trend up + 4h pullback + price above 1d HMA
        # Short: 1d trend down + 4h bounce + price below 1d HMA
        elif chop_trend:
            if hma_1d_slope_bull and price_above_hma_1d:
                if hma_4h_slope_bull and rsi_pullback_long:
                    new_signal = POSITION_SIZE
            elif hma_1d_slope_bear and price_below_hma_1d:
                if hma_4h_slope_bear and rsi_pullback_short:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_1d_slope_bear and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1d_slope_bull and price_above_hma_1d:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals