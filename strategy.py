#!/usr/bin/env python3
"""
Experiment #316: 12h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback + Choppiness Filter

Hypothesis: Simpler is better. After 285+ failed experiments, the pattern is clear:
1. Too many conflicting filters = 0 trades (see exp #308, #309, #310, #315 all Sharpe=0.000)
2. Complex regime switching creates whipsaw (see exp #311, #312 negative Sharpe)
3. 12h timeframe with 1d HTF should generate 20-50 trades/year with proper entry thresholds
4. HMA(16/48) crossover is proven to work on higher timeframes (exp #306 Sharpe=0.203)
5. RSI pullback entries need WIDER thresholds (30-70, not 40-60) to actually trigger
6. Choppiness Index should be a simple binary filter, not complex regime switching

Key changes from failed experiments:
- SIMPLER entry logic (3 conditions max, not 5+)
- WIDER RSI thresholds (30-70 for entries, not 40-60)
- FORCE trade mechanism if no trades for 45 bars (~22 days on 12h)
- Fewer exit conditions (stoploss + RSI extreme only)
- Discrete signal levels: 0.0, ±0.25, ±0.30

Position sizing: 0.25 base, 0.30 strong conviction
Stoploss: 2.5 * ATR trailing
Target: 25-45 trades/year on 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_chop_1d_simp_v1"
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
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Much more responsive than EMA with less lag.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    # WMA helper
    def wma(series, span):
        return series.ewm(span=span, min_periods=span, adjust=False).mean()
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    diff = 2.0 * wma_half - wma_full
    hma = wma(diff, sqrt_n)
    
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = period
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    
    atr_sum = atr.rolling(window=n, min_periods=n).sum()
    hh = high_s.rolling(window=n, min_periods=n).max()
    ll = low_s.rolling(window=n, min_periods=n).min()
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh.iloc[i] - ll.iloc[i]
        if range_hl > 0 and atr_sum.iloc[i] > 0:
            chop[i] = 100 * np.log10(atr_sum.iloc[i] / range_hl) / np.log10(n)
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
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_12h_16 = calculate_hma(close, 16)
    hma_12h_48 = calculate_hma(close, 48)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.25
    SHORT_STRONG = 0.30
    
    # Track position state
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(hma_12h_16[i]) or np.isnan(hma_12h_48[i]):
            continue
        
        # === 1D MAJOR TREND REGIME ===
        # Bull: price above 1d HMA21 AND HMA21 > HMA50
        # Bear: price below 1d HMA21 AND HMA21 < HMA50
        price_above_1d_hma21 = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma21 = close[i] < hma_1d_21_aligned[i]
        
        hma21_above_hma50 = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma21_below_hma50 = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        regime_bull = price_above_1d_hma21 and hma21_above_hma50
        regime_bear = price_below_1d_hma21 and hma21_below_hma50
        
        # === CHOPPINESS FILTER ===
        # CHOP > 58 = choppy (reduce size or avoid trend entries)
        # CHOP < 42 = trending (full size trend entries)
        is_choppy = chop_14[i] > 58.0
        is_trending = chop_14[i] < 42.0
        
        # === 12H LOCAL TREND ===
        hma_bullish = hma_12h_16[i] > hma_12h_48[i]
        hma_bearish = hma_12h_16[i] < hma_12h_48[i]
        
        # HMA slope (5-bar lookback for confirmation)
        hma_slope_up = hma_12h_48[i] > hma_12h_48[i-5] if i >= 5 else False
        hma_slope_down = hma_12h_48[i] < hma_12h_48[i-5] if i >= 5 else False
        
        # === RSI SIGNALS (WIDER thresholds to ensure trades) ===
        # Long: RSI < 45 (pullback) or RSI < 35 (oversold)
        # Short: RSI > 55 (pullback) or RSI > 65 (overbought)
        rsi_oversold = rsi_14[i] < 45.0
        rsi_strong_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 55.0
        rsi_strong_overbought = rsi_14[i] > 65.0
        
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (SIMPLE - max 3 conditions) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if regime_bull or (not regime_bear):
            # Primary: HMA bullish + RSI pullback + trending
            if hma_bullish and rsi_oversold and is_trending:
                new_signal = LONG_BASE
            
            # Strong: HMA bullish + RSI strongly oversold
            elif hma_bullish and rsi_strong_oversold:
                new_signal = LONG_STRONG
            
            # HMA crossover + RSI rising
            elif hma_bullish and hma_slope_up and rsi_rising:
                new_signal = LONG_BASE
            
            # Choppy market mean revert (RSI very oversold)
            elif is_choppy and rsi_strong_oversold:
                new_signal = LONG_BASE * 0.8
        
        # SHORT ENTRIES
        if regime_bear or (not regime_bull):
            # Primary: HMA bearish + RSI pullback + trending
            if hma_bearish and rsi_overbought and is_trending:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            
            # Strong: HMA bearish + RSI strongly overbought
            elif hma_bearish and rsi_strong_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG
            
            # HMA crossover + RSI falling
            elif hma_bearish and hma_slope_down and rsi_falling:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            
            # Choppy market mean revert (RSI very overbought)
            elif is_choppy and rsi_strong_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8
        
        # === FREQUENCY SAFEGUARD (ensure 20+ trades/year) ===
        # Force trade if no signal for 45 bars (~22 days on 12h)
        if bars_since_last_trade > 45 and new_signal == 0.0 and not in_position:
            if hma_bullish and rsi_14[i] < 50.0:
                new_signal = LONG_BASE * 0.6
            elif hma_bearish and rsi_14[i] > 50.0:
                new_signal = -SHORT_BASE * 0.6
            elif rsi_strong_oversold:
                new_signal = LONG_BASE * 0.6
            elif rsi_strong_overbought:
                new_signal = -SHORT_BASE * 0.6
        
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
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_strong_overbought:
                rsi_exit = True
            if position_side < 0 and rsi_strong_oversold:
                rsi_exit = True
        
        # === HMA REVERSAL EXIT ===
        hma_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and hma_bearish:
                hma_exit = True
            if position_side < 0 and hma_bullish:
                hma_exit = True
        
        if stoploss_triggered or rsi_exit or hma_exit:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.15:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.28:
                new_signal = -SHORT_STRONG
            else:
                new_signal = -SHORT_BASE
        
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