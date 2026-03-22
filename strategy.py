#!/usr/bin/env python3
"""
Experiment #376: 12h Primary + 1d HTF — Simplified Trend-Follow with RSI Pullback

Hypothesis: After 350+ experiments, complex dual-regime strategies keep failing.
The winning pattern is SIMPLICITY:
1. 1d HMA(21) for major trend direction (proven in current best strategy)
2. 12h RSI(14) for pullback entries in trend direction
3. Choppiness Index for POSITION SIZING only (not regime switching)
4. ATR(14) trailing stop at 2.5x for risk management
5. Fewer filters = more trades = better statistical significance

Why this might beat Sharpe=0.435:
- Simpler entry logic avoids the "too many filters = 0 trades" problem
- 12h TF generates 25-45 trades/year (optimal frequency)
- 1d trend filter prevents counter-trend trades (major lesson from 2022 crash)
- RSI pullback entries catch dips in uptrends, rallies in downtrends
- Choppiness adjusts size (smaller in chop, larger in trend) without switching logic

Position sizing: 0.25 base, 0.30 strong trend, 0.15 choppy
Stoploss: 2.5 * ATR trailing
Target: 30-50 trades/year on 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_simp_trend_rsi_chop_1d_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/ranging market
    CHOP < 38.2 = trending market
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    chop[np.isnan(chop)] = 50.0
    
    return chop

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    sma_50 = calculate_sma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    SIZE_TREND = 0.30    # Strong trend (CHOP < 38.2)
    SIZE_NORMAL = 0.25   # Normal conditions
    SIZE_CHOP = 0.15     # Choppy market (CHOP > 61.8)
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(sma_50[i]):
            continue
        
        # === 1D MAJOR TREND DIRECTION ===
        # Price above 1d HMA = bullish bias (favor longs)
        # Price below 1d HMA = bearish bias (favor shorts)
        trend_bull = close[i] > hma_1d_21_aligned[i]
        trend_bear = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS FOR POSITION SIZING ===
        choppy = chop_14[i] > 61.8
        trending = chop_14[i] < 38.2
        
        if trending:
            base_size = SIZE_TREND
        elif choppy:
            base_size = SIZE_CHOP
        else:
            base_size = SIZE_NORMAL
        
        # === RSI PULLBACK ENTRY LOGIC ===
        # Long: Trend bull + RSI pullback to 35-50 zone
        # Short: Trend bear + RSI rally to 50-65 zone
        
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_neutral_long = 35.0 < rsi_14[i] < 50.0
        rsi_neutral_short = 50.0 < rsi_14[i] < 65.0
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        # Long entry: Bull trend + RSI pullback
        if trend_bull and rsi_neutral_long:
            new_signal = base_size
        
        # Strong long: Bull trend + RSI oversold (deep pullback)
        if trend_bull and rsi_oversold:
            new_signal = SIZE_TREND
        
        # Short entry: Bear trend + RSI rally
        if trend_bear and rsi_neutral_short:
            if new_signal == 0.0:
                new_signal = -base_size
        
        # Strong short: Bear trend + RSI overbought (sharp rally)
        if trend_bear and rsi_overbought:
            if new_signal == 0.0:
                new_signal = -SIZE_TREND
        
        # === FREQUENCY BOOST (ensure 30+ trades/year) ===
        # If no trade for 15 bars (~7.5 days on 12h), force entry on weaker signal
        bars_since_trade = i - last_trade_bar
        if bars_since_trade > 15 and new_signal == 0.0 and not in_position:
            if trend_bull and rsi_14[i] < 50.0:
                new_signal = SIZE_NORMAL * 0.6
            elif trend_bear and rsi_14[i] > 50.0:
                new_signal = -SIZE_NORMAL * 0.6
        
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
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_bear:
                trend_reversal = True
            if position_side < 0 and trend_bull:
                trend_reversal = True
        
        # === RSI EXTREME EXIT (take profit) ===
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_14[i] > 75.0:
                rsi_exit = True
            if position_side < 0 and rsi_14[i] < 25.0:
                rsi_exit = True
        
        if stoploss_triggered or trend_reversal or rsi_exit:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if new_signal > 0.28:
                new_signal = SIZE_TREND
            elif new_signal > 0.18:
                new_signal = SIZE_NORMAL
            elif new_signal > 0:
                new_signal = SIZE_CHOP
            elif new_signal < -0.28:
                new_signal = -SIZE_TREND
            elif new_signal < -0.18:
                new_signal = -SIZE_NORMAL
            else:
                new_signal = -SIZE_CHOP
        
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