#!/usr/bin/env python3
"""
Experiment #416: 12h Primary + 1d HTF — Dual Regime HMA + RSI + ATR

Hypothesis: 12h timeframe reduces whipsaw noise compared to 4h while maintaining
adequate trade frequency. Key innovations vs previous attempts:
1. Dual regime logic: CHOP > 61.8 = mean reversion, CHOP < 38.2 = trend follow
2. HMA(21/63) crossover — smoother than 16/48 for 12h TF
3. RSI(14) with asymmetric thresholds (long <35, short >65) — proven on ETH
4. 1d HTF HMA(21) for overall market bias filter
5. ATR(14) trailing stoploss at 3.0x ATR — wider stops for 12h volatility
6. Position sizing: 0.30 base, reduced to 0.15 on vol spikes
7. Simplified hold logic to maintain positions through minor fluctuations

Why this should beat Sharpe=0.612:
- 12h has fewer false signals than 4h (less noise)
- Dual regime adapts to market conditions (range vs trend)
- 1d HTF filter prevents counter-trend trades in strong moves
- Conservative sizing (0.30 max) limits drawdown during 2022 crash
- Target: 30-50 trades/year on 12h (2-4 per month)

Target: Sharpe > 0.612, 120-200 trades over 4-year train, DD < -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_hma_rsi_dual_regime_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest - lowest < 1e-10:
            chop[i] = 50.0
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            tr_sum += tr
        
        chop[i] = 100.0 * np.log10(tr_sum / (highest - lowest)) / np.log10(period)
        chop[i] = np.clip(chop[i], 0, 100)
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    hma_21 = calculate_hma(close, 21)
    hma_63 = calculate_hma(close, 63)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF HMA for bias (1d)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[100:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 12h
    REDUCED_SIZE = 0.15  # Half size on vol spikes
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(chop[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_21[i]) or np.isnan(hma_63[i]):
            continue
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # === HTF BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (12h HMA crossover) ===
        hma_bullish = hma_21[i] > hma_63[i]
        hma_bearish = hma_21[i] < hma_63[i]
        
        # === RSI THRESHOLDS (asymmetric) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.0:
            position_size = REDUCED_SIZE
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP
        if price_above_hma_1d:  # HTF bullish bias required for longs
            if is_trending and hma_bullish:
                # Trend following mode: enter on pullback
                if rsi_14[i] < 55.0:  # Pullback in uptrend
                    desired_signal = position_size
            elif is_choppy:
                # Mean reversion in range: buy oversold
                if rsi_oversold:
                    desired_signal = position_size
            elif hma_bullish:
                # HMA bullish, wait for pullback
                if rsi_14[i] < 50.0:
                    desired_signal = position_size
        
        # SHORT SETUP
        if price_below_hma_1d:  # HTF bearish bias required for shorts
            if is_trending and hma_bearish:
                # Trend following mode: enter on rally
                if rsi_14[i] > 45.0:  # Rally in downtrend
                    desired_signal = -position_size
            elif is_choppy:
                # Mean reversion in range: sell overbought
                if rsi_overbought:
                    desired_signal = -position_size
            elif hma_bearish:
                # HMA bearish, wait for rally
                if rsi_14[i] > 50.0:
                    desired_signal = -position_size
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT ===
        if in_position and position_side > 0 and rsi_14[i] > 80.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 20.0:
            desired_signal = 0.0
        
        # === HTF BIAS REVERSAL EXIT ===
        if in_position and position_side > 0 and price_below_hma_1d:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1d:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_1d:
                desired_signal = position_size
            elif position_side < 0 and price_below_hma_1d:
                desired_signal = -position_size
        
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