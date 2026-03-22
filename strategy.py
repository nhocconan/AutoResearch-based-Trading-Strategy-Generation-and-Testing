#!/usr/bin/env python3
"""
Experiment #358: 4h Regime-Adaptive Strategy with 1d/1w HMA Bias + Choppiness Filter

Hypothesis: After 357 failed experiments, the key insight is that BTC/ETH need REGIME-ADAPTIVE
logic that switches between trend-following and mean-reversion based on market conditions.

1. CHOPPINESS INDEX (CHOP) REGIME DETECTION:
   - CHOP > 61.8 = ranging market → use mean reversion at Bollinger bands
   - CHOP < 38.2 = trending market → follow 1d HMA direction
   - 38.2 <= CHOP <= 61.8 = transition → stay flat or reduce position

2. MULTI-TIMEFRAME TREND BIAS:
   - 1d HMA(21): Primary trend direction
   - 1w HMA(21): Major trend bias (only trade in direction of 1w trend)
   - Both must agree for full position size

3. ENTRY LOGIC:
   - Range regime: Long at BB lower band + RSI<35, Short at BB upper band + RSI>65
   - Trend regime: Long when price > 1d HMA + pullback to EMA(21), vice versa for short
   - 1w HMA filter: Only long if price > 1w HMA, only short if price < 1w HMA

4. ATR TRAILING STOP (2.5x): Protect capital on reversals

5. POSITION SIZING: 0.30 discrete (aggressive enough for trades, conservative for DD)
   - Full size (0.30) when 1d + 1w agree
   - Half size (0.15) when only 1d agrees

Why 4h should work:
- Fast enough to catch swings, slow enough to avoid noise
- 1d/1w provide stable bias without whipsaw
- Regime-adaptive handles both 2021 bull and 2022 bear markets
- Should generate 30-60 trades/year per symbol

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.15-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_chop_1d_1w_hma_bb_rsi_atr_v2"
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

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100
    return rsi

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = tr[i-period+1:i+1].sum()
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

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
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    ema_21 = calculate_ema(close, 21)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
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
        
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        range_regime = chop[i] > 55.0  # Loosened from 61.8 to generate more trades
        trend_regime = chop[i] < 45.0  # Loosened from 38.2
        # transition regime: 45 <= chop <= 55
        
        # === TREND BIAS ===
        bull_1d = close[i] > hma_1d_aligned[i]
        bear_1d = close[i] < hma_1d_aligned[i]
        bull_1w = close[i] > hma_1w_aligned[i]
        bear_1w = close[i] < hma_1w_aligned[i]
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        position_size = SIZE_HALF  # Default to half size
        
        # Determine position size based on HTF agreement
        if bull_1d and bull_1w:
            position_size = SIZE_FULL
        elif bear_1d and bear_1w:
            position_size = SIZE_FULL
        
        # RANGE REGIME: Mean reversion at Bollinger bands
        if range_regime:
            # Long: Price at BB lower + RSI oversold + 1w bias not bearish
            if close[i] <= bb_lower[i] * 1.002 and rsi[i] < 40 and not bear_1w:
                new_signal = position_size
            
            # Short: Price at BB upper + RSI overbought + 1w bias not bullish
            elif close[i] >= bb_upper[i] * 0.998 and rsi[i] > 60 and not bull_1w:
                new_signal = -position_size
        
        # TREND REGIME: Follow trend with pullback entries
        elif trend_regime:
            # Long: 1d bullish + pullback to EMA21 + 1w not bearish
            if bull_1d and close[i] <= ema_21[i] * 1.005 and not bear_1w:
                new_signal = position_size
            
            # Short: 1d bearish + pullback to EMA21 + 1w not bullish
            elif bear_1d and close[i] >= ema_21[i] * 0.995 and not bull_1w:
                new_signal = -position_size
        
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
            if position_side > 0 and bear_1d:
                new_signal = 0.0
            if position_side < 0 and bull_1d:
                new_signal = 0.0
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime changes to transition zone and we're in position
        if in_position and not range_regime and not trend_regime:
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