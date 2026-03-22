#!/usr/bin/env python3
"""
Experiment #352: 4h Z-Score Mean Reversion with 1d/1w HMA Trend Filter

Hypothesis: After 300+ failed strategies, the key insight is that complex regime
detection often fails due to overfitting. Instead, use a simpler but robust approach:

1. Z-SCORE EXTREMES for entry signals:
   - Z-score = (price - SMA20) / StdDev(20)
   - Long when Z < -1.5 (oversold extreme)
   - Short when Z > +1.5 (overbought extreme)
   - This captures mean reversion opportunities in both bull and bear markets

2. 1D HMA for trend bias (proven stable filter):
   - Only long if price > 1d HMA (bullish macro bias)
   - Only short if price < 1d HMA (bearish macro bias)
   - HMA smoother than EMA, less whipsaw on 1d timeframe

3. 1W HMA for macro regime filter:
   - If price > 1w HMA = bull macro (favor longs, reduce short size)
   - If price < 1w HMA = bear macro (favor shorts, reduce long size)
   - Asymmetric sizing based on macro regime

4. ATR(14) stoploss at 2.5x for risk management

5. RSI(14) momentum confirmation:
   - Long: RSI < 45 (not overbought on entry)
   - Short: RSI > 55 (not oversold on entry)

Why 4h timeframe:
- Slow enough to avoid noise and fee churn
- Fast enough to generate 10+ trades per symbol per year
- Works well with 1d/1w HTF alignment

Position sizing: 0.25 base, asymmetric based on 1w HMA regime
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_zscore_1d_1w_hma_asymmetric_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / np.maximum(rolling_std, 1e-10)
    return zscore.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    zscore = calculate_zscore(close, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels with asymmetry (Rule 4)
    SIZE_BASE = 0.25
    SIZE_LONG = 0.30  # Slightly larger in bull macro
    SIZE_SHORT = 0.30  # Slightly larger in bear macro
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(zscore[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === 1D HMA TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === 1W HMA MACRO REGIME ===
        bull_macro_1w = close[i] > hma_1w_aligned[i]
        bear_macro_1w = close[i] < hma_1w_aligned[i]
        
        # === Z-SCORE EXTREMES ===
        z_oversold = zscore[i] < -1.5  # Loosened for more trades
        z_overbought = zscore[i] > 1.5  # Loosened for more trades
        
        # === RSI MOMENTUM FILTER ===
        rsi_not_overbought = rsi[i] < 55  # Allow longs when RSI not too high
        rsi_not_oversold = rsi[i] > 45  # Allow shorts when RSI not too low
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: Z-score oversold + 1d HMA bullish + RSI confirmation
        if z_oversold and bull_trend_1d and rsi_not_overbought:
            # Size based on 1w macro regime
            if bull_macro_1w:
                new_signal = SIZE_LONG  # Full size in bull macro
            else:
                new_signal = SIZE_BASE  # Reduced size in bear macro
        
        # SHORT ENTRY: Z-score overbought + 1d HMA bearish + RSI confirmation
        elif z_overbought and bear_trend_1d and rsi_not_oversold:
            # Size based on 1w macro regime
            if bear_macro_1w:
                new_signal = -SIZE_SHORT  # Full size in bear macro
            else:
                new_signal = -SIZE_BASE  # Reduced size in bull macro
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 1d trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0
        
        # === Z-SCORE MEAN REVERSION EXIT ===
        # Exit long when Z-score becomes positive (mean reached)
        if in_position and position_side > 0 and zscore[i] > 0.5:
            new_signal = 0.0
        
        # Exit short when Z-score becomes negative (mean reached)
        if in_position and position_side < 0 and zscore[i] < -0.5:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals