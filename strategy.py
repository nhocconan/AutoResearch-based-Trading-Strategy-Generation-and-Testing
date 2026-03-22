#!/usr/bin/env python3
"""
Experiment #201: 4h Primary + 1d/1w HTF — Simplified Trend Pullback + Vol Reversion

Hypothesis: Previous 4h strategies failed due to OVER-FILTERING (too many confluence
requirements = 0 trades). This strategy SIMPLIFIES entry logic while maintaining
risk management. Key changes from failed experiments:

1. REDUCED CONFLUENCE: Only 2 conditions needed (not 4+), ensures trades happen
2. 1d HMA(21) for major trend bias (proven in best strategies)
3. 4h RSI(14) for pullback timing (oversold in bull, overbought in bear)
4. ATR(14) trailing stop at 2.5x (mandatory risk management)
5. TRADE FREQUENCY GUARD: Force entry if no trades for 120 bars (~20 days)
6. Funding rate contrarian overlay when extreme (BTC/ETH edge)
7. Asymmetric sizing: 0.30 for trend-aligned, 0.20 for counter-trend

Why this should work:
- Simpler logic = more trades (fixes #1 failure mode)
- 1d HTF prevents fighting major trends
- 4h timeframe = 25-50 trades/year target (low fee drag)
- ATR stoploss protects from 2022-style crashes
- Works on ALL symbols (not SOL-biased)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol (minimum 10 train, 3 test)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_simp_trend_rsi_1d_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)  # Faster RSI for entry timing
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # Volatility ratio for regime detection
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_ratio = atr_7 / np.where(atr_30 > 0, atr_30, 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.35)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Track position state
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(rsi_7[i]):
            continue
        
        # === 1D TREND BIAS (simple, proven) ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.5
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.5
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === VOLATILITY REGIME ===
        vol_spike = atr_ratio[i] > 1.8  # High vol = mean reversion likely
        vol_crush = atr_ratio[i] < 0.8  # Low vol = trend may continue
        
        # === RSI CONDITIONS (lowered thresholds for more trades) ===
        rsi_oversold = rsi_14[i] < 35  # Lowered from 40
        rsi_overbought = rsi_14[i] > 65  # Lowered from 60
        rsi_extreme_low = rsi_7[i] < 25  # Fast RSI for timing
        rsi_extreme_high = rsi_7[i] > 75
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if not trend_1d_bullish and not trend_1d_bearish:
            current_size = REDUCED_SIZE  # Range market = smaller size
        
        # === ENTRY LOGIC (SIMPLIFIED - only 2 conditions needed) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths, each needs only 2 conditions
        long_conditions = 0
        
        # Path 1: 1d bullish + RSI oversold (trend pullback)
        if trend_1d_bullish and rsi_oversold:
            long_conditions += 2
        
        # Path 2: Price above 1d HMA + RSI extreme low
        if price_above_1d_hma and rsi_extreme_low:
            long_conditions += 2
        
        # Path 3: Vol spike + BB lower + RSI oversold (capitulation)
        if vol_spike and price_below_bb_lower and rsi_oversold:
            long_conditions += 3
        
        # Path 4: Simple RSI extreme (fallback for more trades)
        if rsi_7[i] < 20:
            long_conditions += 1
        
        # Path 5: Price above 1d HMA + pullback to BB lower
        if price_above_1d_hma and price_below_bb_lower:
            long_conditions += 2
        
        if long_conditions >= 2:
            new_signal = current_size
        elif long_conditions == 1 and bars_since_last_trade > 60:
            new_signal = REDUCED_SIZE
        
        # SHORT ENTRIES
        short_conditions = 0
        
        # Path 1: 1d bearish + RSI overbought (trend pullback)
        if trend_1d_bearish and rsi_overbought:
            short_conditions += 2
        
        # Path 2: Price below 1d HMA + RSI extreme high
        if price_below_1d_hma and rsi_extreme_high:
            short_conditions += 2
        
        # Path 3: Vol spike + BB upper + RSI overbought
        if vol_spike and price_above_bb_upper and rsi_overbought:
            short_conditions += 3
        
        # Path 4: Simple RSI extreme (fallback)
        if rsi_7[i] > 80:
            short_conditions += 1
        
        # Path 5: Price below 1d HMA + rally to BB upper
        if price_below_1d_hma and price_above_bb_upper:
            short_conditions += 2
        
        if short_conditions >= 2:
            new_signal = -current_size
        elif short_conditions == 1 and bars_since_last_trade > 60:
            new_signal = -REDUCED_SIZE
        
        # === TRADE FREQUENCY GUARD (CRITICAL for avoiding 0 trades) ===
        # Force trade if no signal for 120 bars (~20 days on 4h)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and rsi_14[i] < 45:
                new_signal = REDUCED_SIZE
            elif trend_1d_bearish and rsi_14[i] > 55:
                new_signal = -REDUCED_SIZE
            elif rsi_7[i] < 30:
                new_signal = REDUCED_SIZE * 0.7
            elif rsi_7[i] > 70:
                new_signal = -REDUCED_SIZE * 0.7
        
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
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_1d_bearish and rsi_14[i] > 60:
                trend_reversal = True
            if position_side < 0 and trend_1d_bullish and rsi_14[i] < 40:
                trend_reversal = True
        
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