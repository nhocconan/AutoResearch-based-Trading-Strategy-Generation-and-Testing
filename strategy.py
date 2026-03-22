#!/usr/bin/env python3
"""
Experiment #236: 12h Primary + 1d HTF — Simplified Trend/Mean Reversion Hybrid

Hypothesis: After 235 experiments, complexity is the enemy. Strategies with too many
conflicting filters generate 0 trades. The key is SIMPLER logic with LOOSER thresholds.

This strategy uses:
1. 1d HMA(21) for PRIMARY trend direction (bull if price > HMA, bear if price < HMA)
2. 12h Choppiness Index(14) to detect range vs trend (CHOP > 55 = range, < 45 = trend)
3. 12h RSI(14) for entry timing (35/65 thresholds - looser than typical 30/70)
4. 12h ATR(14) for stoploss (2.0x trailing stop)
5. Force-trade every 30 bars to guarantee minimum trade frequency

Key improvements over failed experiments:
- FEWER AND conditions (each filter eliminates trades)
- LOOSER RSI thresholds (35/65 not 30/70) for guaranteed trade frequency
- Simpler regime detection (price vs HMA, not slope calculations)
- Force-trade after 30 bars of no signal (not 40-60)
- 12h primary timeframe = naturally fewer trades = less fee drag

Position sizing: 0.25 base, 0.30 strong signals (discrete levels)
Target: 25-45 trades/year per symbol (within 12h cost model)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_simp_trend_chop_rsi_1d_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (primary trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    hma_12h_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -30
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        # === 1D TREND DIRECTION (simple price vs HMA) ===
        # Bull: price above 1d HMA
        # Bear: price below 1d HMA
        daily_bull = close[i] > hma_1d_21_aligned[i]
        daily_bear = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert)
        # CHOP < 45 = trend market (trend follow)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === 12H LOCAL SIGNALS ===
        price_above_12h_hma = close[i] > hma_12h_21[i]
        price_below_12h_hma = close[i] < hma_12h_21[i]
        
        # === RSI MOMENTUM (LOOSE THRESHOLDS) ===
        rsi_oversold = rsi_14[i] < 35  # Mean reversion long
        rsi_overbought = rsi_14[i] > 65  # Mean reversion short
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        
        # === POSITION SIZING ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND FOLLOWING MODE (when trending + daily trend aligned)
        if is_trending:
            # LONG: Daily bull + price above 12h HMA + RSI bullish
            if daily_bull and price_above_12h_hma and rsi_bullish:
                new_signal = STRONG_SIZE
            # LONG: Daily bull + RSI > 45 (pullback entry)
            elif daily_bull and rsi_14[i] > 45 and rsi_14[i] < 60:
                new_signal = BASE_SIZE
            
            # SHORT: Daily bear + price below 12h HMA + RSI bearish
            if daily_bear and price_below_12h_hma and rsi_bearish:
                new_signal = -STRONG_SIZE
            # SHORT: Daily bear + RSI < 55 (pullback entry)
            elif daily_bear and rsi_14[i] < 55 and rsi_14[i] > 40:
                new_signal = -BASE_SIZE
        
        # MEAN REVERSION MODE (when choppy)
        if is_choppy:
            # LONG: RSI oversold in choppy market
            if rsi_oversold:
                new_signal = BASE_SIZE
            # SHORT: RSI overbought in choppy market
            elif rsi_overbought:
                new_signal = -BASE_SIZE
        
        # === NEUTRAL REGIME (between choppy and trending) ===
        if not is_choppy and not is_trending:
            # Follow daily trend with RSI filter
            if daily_bull and rsi_bullish and price_above_12h_hma:
                new_signal = BASE_SIZE
            elif daily_bear and rsi_bearish and price_below_12h_hma:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 30 bars (~15 days on 12h)
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if daily_bull and rsi_14[i] > 45:
                new_signal = BASE_SIZE * 0.5
            elif daily_bear and rsi_14[i] < 55:
                new_signal = -BASE_SIZE * 0.5
            elif rsi_oversold:
                new_signal = BASE_SIZE * 0.4
            elif rsi_overbought:
                new_signal = -BASE_SIZE * 0.4
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but daily trend turns bearish
            if position_side > 0 and daily_bear and price_below_12h_hma:
                regime_reversal = True
            # Short position but daily trend turns bullish
            if position_side < 0 and daily_bull and price_above_12h_hma:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
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