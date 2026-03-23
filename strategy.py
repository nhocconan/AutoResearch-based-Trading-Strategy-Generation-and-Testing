#!/usr/bin/env python3
"""
Experiment #363: 1d Primary + 1w HTF — HMA Trend with RSI Pullback Entries

Hypothesis: Previous 1d strategies failed because:
1. CRSI+Donchian combinations were too complex and rarely triggered
2. Too many regime filters prevented trades (0 trades = auto-reject)
3. Need simpler, proven patterns: HMA trend + RSI pullback works on 4h, should work on 1d

This strategy uses:
1. 1w HMA(21) as MACRO BIAS (only long if price > 1w HMA, only short if price < 1w HMA)
2. 1d HMA(16/48) crossover for trend direction
3. RSI(14) pullback entries (RSI<45 for long, RSI>55 for short) - RELAXED thresholds
4. Choppiness Index > 55 = reduce position size by 50% (choppy = smaller bets)
5. ATR(14) trailing stop at 2.5x for risk management
6. Position size: 0.30 normal, 0.15 in choppy regimes

KEY INSIGHT: Simpler is better. HMA trend + RSI pullback is proven on 4h.
On 1d, this should generate 15-30 trades/year with clean trend following.
1w HMA bias prevents trading against the macro trend (critical for 2022 crash).

TARGET: 15-30 trades/year on 1d, Sharpe > 0.6 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_1w_bias_chop_size_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

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
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # HMA for trend detection (fast and slow)
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    
    # Calculate and align 1w HMA for macro bias (HARD FILTER)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 1d (target 15-30 trades/year)
    CHOP_SIZE = 0.15  # 15% in choppy regimes
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1w HMA - HARD FILTER) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # High choppiness = reduce size
        is_trending = chop[i] <= 55.0  # Low choppiness = full size
        
        # Select position size based on regime
        current_size = CHOP_SIZE if is_choppy else BASE_SIZE
        
        # === TREND DIRECTION (HMA crossover) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === RSI PULLBACK ENTRY (RELAXED thresholds for trade frequency) ===
        rsi_oversold = rsi_14[i] < 45.0  # Long entry on pullback
        rsi_overbought = rsi_14[i] > 55.0  # Short entry on rally
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG: 1w bullish + 1d bullish + RSI pullback
        if price_above_hma_1w and hma_bullish and rsi_oversold:
            desired_signal = current_size
        
        # SHORT: 1w bearish + 1d bearish + RSI rally
        elif price_below_hma_1w and hma_bearish and rsi_overbought:
            desired_signal = -current_size
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND EXIT (HMA crossover against position) ===
        if in_position and position_side > 0 and hma_bearish:
            # Long position: exit when HMA turns bearish
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_bullish:
            # Short position: exit when HMA turns bullish
            desired_signal = 0.0
        
        # === MACRO BIAS EXIT (1w HMA against position) ===
        if in_position and position_side > 0 and price_below_hma_1w:
            # Long position: exit when macro turns bearish
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1w:
            # Short position: exit when macro turns bullish
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 70:
            # Long position: take profit at RSI overbought
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 30:
            # Short position: take profit at RSI oversold
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Check if trend and bias still valid
            if position_side > 0:
                if price_above_hma_1w and hma_bullish:
                    desired_signal = current_size
            elif position_side < 0:
                if price_below_hma_1w and hma_bearish:
                    desired_signal = -current_size
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals