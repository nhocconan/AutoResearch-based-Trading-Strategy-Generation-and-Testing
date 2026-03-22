#!/usr/bin/env python3
"""
Experiment #093: 1d Primary + 1w HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Complex regime-switching strategies (choppiness, dual-regime) have failed
repeatedly in experiments #081-#092. Simpler trend-following with clean entry timing
works better on 1d timeframe.

Key insights from research:
1. HMA(21/50) crossover provides cleaner trend signals than EMA (less lag)
2. RSI(14) pullback to 35-55 in uptrend = high-probability long entry
3. 1w HMA slope = major trend filter (only trade with weekly trend)
4. ATR(14) 2.5x trailing stop protects capital during 2022-style crashes
5. Position size 0.25-0.30 discrete minimizes fee churn while capturing moves

Why 1d works for BTC/ETH:
- Natural trade frequency: 20-40 trades/year (avoids fee drag)
- Less noise than lower TFs (fewer whipsaws in 2022 bear)
- Captures major crypto moves (2021 bull, 2022 bear, 2023-24 recovery)
- 1w HTF prevents counter-trend trades in major reversals

Timeframe: 1d (REQUIRED per experiment #093)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-40/year per symbol (must exceed 10 on train, 3 on test)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_pullback_1w_v1"
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

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change over lookback."""
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    last_entry_bar = -100
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            signals[i] = 0.0
            continue
        
        # === 1W TREND BIAS (MAJOR) ===
        # HMA slope > 0.5 = bullish bias (prefer longs)
        # HMA slope < -0.5 = bearish bias (prefer shorts)
        trend_1w_bullish = hma_1w_slope_aligned[i] > 0.5
        trend_1w_bearish = hma_1w_slope_aligned[i] < -0.5
        
        # Price vs 1w HMA for additional confirmation
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === 1D TREND CONFIRMATION ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === RSI PULLBACK ENTRIES ===
        # Long: RSI pulled back to 35-55 in uptrend
        # Short: RSI rallied to 45-65 in downtrend
        rsi_pullback_long = 32 < rsi_14[i] < 58
        rsi_pullback_short = 42 < rsi_14[i] < 68
        
        # Strong RSI signals for mean reversion
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_entry = i - last_entry_bar
        
        # LONG ENTRIES (primary: with 1w trend)
        if trend_1w_bullish and hma_bullish and rsi_pullback_long:
            new_signal = BASE_SIZE
        elif trend_1w_bullish and price_above_1w_hma and rsi_14[i] < 50:
            new_signal = BASE_SIZE * 0.9
        elif hma_bullish and rsi_oversold and price_above_1w_hma:
            # Strong oversold with price above weekly HMA
            new_signal = BASE_SIZE * 0.8
        elif hma_bullish and rsi_14[i] < 40:
            # Deep pullback in daily uptrend
            new_signal = BASE_SIZE * 0.7
        
        # SHORT ENTRIES (primary: with 1w trend)
        if trend_1w_bearish and hma_bearish and rsi_pullback_short:
            new_signal = -BASE_SIZE
        elif trend_1w_bearish and price_below_1w_hma and rsi_14[i] > 50:
            new_signal = -BASE_SIZE * 0.9
        elif hma_bearish and rsi_overbought and price_below_1w_hma:
            # Strong overbought with price below weekly HMA
            new_signal = -BASE_SIZE * 0.8
        elif hma_bearish and rsi_14[i] > 60:
            # Strong rally in daily downtrend
            new_signal = -BASE_SIZE * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 90 bars (~90 days on 1d), allow weaker entry
        if bars_since_entry > 90 and new_signal == 0.0 and not in_position:
            if trend_1w_bullish and rsi_14[i] < 45:
                new_signal = BASE_SIZE * 0.5
            elif trend_1w_bearish and rsi_14[i] > 55:
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
        
        # Apply stoploss
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_entry_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals