#!/usr/bin/env python3
"""
Experiment #381: 4h Primary + 1d HTF — Simplified Trend with Choppiness Scaling

Hypothesis: After analyzing 380+ experiments, the pattern is clear:
1. Dual-regime switching creates whipsaws (see #374, #379 failures)
2. Simpler trend-follow with pullback entries works best (current best: Sharpe=0.435)
3. Choppiness Index should SCALE position size, not SWITCH strategies
4. 4h timeframe with 1d HTF is proven (like current best but faster TF)
5. RSI pullback entries in direction of 1d HMA trend
6. When CHOP > 61.8: reduce size to 0.15 (choppy = smaller bets)
7. When CHOP < 38.2: full size 0.30 (trending = conviction)
8. ATR 2.5x trailing stop mandatory

Why this might beat current best (Sharpe=0.435):
- 4h generates more signals than 1d while keeping fee drag low
- Choppiness scaling avoids big losses in range markets
- Simple logic = fewer bugs, more reliable trade generation
- RSI pullback (not breakout) catches better entries in trends

Position sizing: 0.15-0.30 based on choppiness
Stoploss: 2.5 * ATR trailing
Target: 30-50 trades/year on 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trend_rsi_chop_scale_1d_v1"
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
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
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
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_8 = calculate_hma(close, period=8)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Scaled by choppiness: trending = full size, choppy = half size
    SIZE_TREND = 0.30
    SIZE_CHOP = 0.15
    
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
        
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_8[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        # Bull: price above 1d HMA(21)
        # Bear: price below 1d HMA(21)
        trend_bull = close[i] > hma_1d_21_aligned[i]
        trend_bear = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS (position size scaler, NOT regime switch) ===
        # CHOP > 61.8 = choppy (reduce size)
        # CHOP < 38.2 = trending (full size)
        # 38.2-61.8 = neutral (medium size)
        is_choppy = chop_14[i] > 61.8
        is_trending = chop_14[i] < 38.2
        
        if is_trending:
            position_size = SIZE_TREND
        elif is_choppy:
            position_size = SIZE_CHOP
        else:
            position_size = (SIZE_TREND + SIZE_CHOP) / 2.0  # 0.225
        
        # === 4H LOCAL TREND CONFIRMATION ===
        hma_bullish = hma_4h_8[i] > hma_4h_21[i]
        hma_bearish = hma_4h_8[i] < hma_4h_21[i]
        
        # === RSI PULLBACK ENTRY SIGNALS ===
        # Long: 1d trend bull + 4h HMA bull + RSI pullback (35-50)
        # Short: 1d trend bear + 4h HMA bear + RSI pullback (50-65)
        rsi_pullback_long = 35.0 < rsi_14[i] < 50.0
        rsi_pullback_short = 50.0 < rsi_14[i] < 65.0
        
        # RSI extreme (stronger signal but less frequent)
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # Long entry: 1d bull + 4h bull + RSI pullback
        if trend_bull and hma_bullish:
            if rsi_pullback_long or rsi_oversold:
                new_signal = position_size
        
        # Short entry: 1d bear + 4h bear + RSI pullback
        if trend_bear and hma_bearish:
            if rsi_pullback_short or rsi_overbought:
                if new_signal == 0.0:  # Don't override long signal
                    new_signal = -position_size
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 20 bars (~3.3 days on 4h)
        # This ensures we generate enough trades (target 30-50/year)
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if trend_bull and rsi_14[i] < 50.0:
                new_signal = position_size * 0.5
            elif trend_bear and rsi_14[i] > 50.0:
                new_signal = -position_size * 0.5
        
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
        
        # === RSI REVERSAL EXIT ===
        # Exit long when RSI overbought, exit short when RSI oversold
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_14[i] > 70.0:
                rsi_exit = True
            if position_side < 0 and rsi_14[i] < 30.0:
                rsi_exit = True
        
        # === TREND REVERSAL EXIT ===
        # Exit when 1d trend flips against position
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_bear:
                trend_reversal = True
            if position_side < 0 and trend_bull:
                trend_reversal = True
        
        if stoploss_triggered or rsi_exit or trend_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if new_signal > 0:
                if new_signal > 0.25:
                    new_signal = SIZE_TREND if is_trending else (SIZE_CHOP if is_choppy else 0.225)
                else:
                    new_signal = 0.15  # Minimum long size
            else:
                if new_signal < -0.25:
                    new_signal = -SIZE_TREND if is_trending else (-SIZE_CHOP if is_choppy else -0.225)
                else:
                    new_signal = -0.15  # Minimum short size
        
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
            # If same side, keep position (no update needed)
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