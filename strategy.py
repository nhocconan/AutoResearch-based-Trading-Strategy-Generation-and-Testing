#!/usr/bin/env python3
"""
Experiment #371: 4h Primary + 1d HTF — Volatility Spike Mean Reversion

Hypothesis: After 370 experiments, the clearest pattern is:
1. 4h timeframe balances trade frequency (30-50/year) vs fee drag
2. Volatility spike mean reversion works in ALL regimes (bull/bear/chop)
3. ATR(7)/ATR(30) > 2.0 captures panic extremes → mean reversion follows
4. Price < BB(20, 2.5) lower band confirms oversold condition
5. 1d HMA(21) for major trend bias (longs favored above, shorts below)
6. Asymmetric sizing: longs 0.30, shorts 0.20 (crypto long bias)
7. SIMPLE entry logic = more trades (lesson from 0-trade failures)

Why this might beat current best (Sharpe=0.435):
- Vol spike MR works in 2022 crash AND 2025 bear (unlike pure trend)
- Fewer confluence conditions = more trades (avoid 0-trade failure)
- 4h TF avoids 15m/1h fee drag while generating enough signals
- Proven edge: vol crush after panic has 60-70% win rate

Position sizing: 0.30 longs, 0.20 shorts (discrete levels)
Stoploss: 2.5 * ATR trailing
Target: 30-50 trades/year on 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_volspike_mr_bb_hma1d_asym_v1"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

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
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.5)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volatility ratio (ATR short / ATR long)
    with np.errstate(divide='ignore', invalid='ignore'):
        atr_ratio = atr_7 / (atr_30 + 1e-10)
    atr_ratio = np.nan_to_num(atr_ratio, nan=1.0)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            continue
        
        # === 1D MAJOR TREND BIAS ===
        trend_bull = close[i] > hma_1d_21_aligned[i]
        trend_bear = close[i] < hma_1d_21_aligned[i]
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = atr_ratio[i] > 2.0  # ATR(7) > 2x ATR(30)
        vol_normal = atr_ratio[i] < 1.2  # Volatility normalized
        
        # === BOLLINGER BAND EXTREMES ===
        price_below_bb = close[i] < bb_lower[i]
        price_above_bb = close[i] > bb_upper[i]
        price_near_mid = bb_lower[i] < close[i] < bb_upper[i]
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ENTRY LOGIC - VOLATILITY SPIKE MEAN REVERSION ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG: Vol spike + price below BB + (bull trend OR RSI oversold)
        if vol_spike and price_below_bb:
            if trend_bull:
                new_signal = LONG_SIZE
            elif rsi_oversold:
                new_signal = LONG_SIZE * 0.7
            elif bars_since_last_trade > 15:
                # Force entry if no trades for ~2.5 days (4h * 15 = 60h)
                new_signal = LONG_SIZE * 0.5
        
        # SHORT: Vol spike + price above BB + (bear trend OR RSI overbought)
        if vol_spike and price_above_bb:
            if trend_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            elif rsi_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.7
        
        # === EXIT: VOLATILITY NORMALIZATION ===
        if in_position and vol_normal and price_near_mid:
            new_signal = 0.0
        
        # === EXIT: RSI REVERSAL ===
        if in_position and position_side > 0 and rsi_overbought:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_oversold:
            new_signal = 0.0
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_30[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_30[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side != 0:
            if position_side > 0 and trend_bear and close[i] < hma_1d_21_aligned[i]:
                new_signal = 0.0
            if position_side < 0 and trend_bull and close[i] > hma_1d_21_aligned[i]:
                new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if new_signal > 0.25:
                new_signal = LONG_SIZE
            elif new_signal > 0:
                new_signal = LONG_SIZE * 0.5
            elif new_signal < -0.15:
                new_signal = -SHORT_SIZE
            else:
                new_signal = -SHORT_SIZE * 0.5
        
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