#!/usr/bin/env python3
"""
Experiment #364: 4h Primary + 12h/1d HTF — Dual Regime ADX/BB with Pullback Entries

Hypothesis: Previous strategies failed due to:
1. Too many exit conditions cutting winners early
2. Fisher Transform too noisy on 4h timeframe
3. Complex hold logic causing premature position flips

This strategy uses proven regime detection from quantitative literature:
1. ADX(14) > 25 = trending regime (use pullback entries)
2. ADX(14) < 20 = ranging regime (use BB mean reversion)
3. ADX 20-25 = hysteresis zone (hold current position)
4. 12h HMA(21) for macro bias (only long if price > 12h HMA)
5. 1d HMA(21) for secondary confirmation

TREND REGIME entries:
- Long: ADX>25 + price>12h_HMA + pullback to EMA21 + RSI(14)>40
- Short: ADX>25 + price<12h_HMA + rally to EMA21 + RSI(14)<60

RANGE REGIME entries:
- Long: ADX<20 + price<BB_lower + RSI(14)<35 + price>12h_HMA
- Short: ADX<20 + price>BB_upper + RSI(14)>65 + price<12h_HMA

EXIT: Only ATR(14) trailing stop at 2.5x - let winners run
SIZE: 0.30 discrete (30% of capital)

TARGET: 30-50 trades/year on 4h, Sharpe > 0.6 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_adx_bb_dual_regime_12h1d_hma_pullback_v1"
timeframe = "4h"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    plus_dm[(plus_dm <= minus_dm)] = 0
    minus_dm[(minus_dm <= plus_dm)] = 0
    
    # Smoothed values (Wilder's)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(0).values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper.values, lower.values, sma.values

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

def calculate_ema(close, period=21):
    """Calculate EMA."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    ema_21 = calculate_ema(close, period=21)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Calculate and align 12h HMA for macro bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA for secondary confirmation
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 4h (target 30-50 trades/year)
    
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
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(ema_21[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (12h HMA - PRIMARY, 1d HMA - SECONDARY) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (ADX with hysteresis) ===
        adx_value = adx_14[i]
        is_trending = adx_value > 25.0
        is_ranging = adx_value < 20.0
        # ADX 20-25 = hysteresis zone (maintain current position)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND REGIME: Pullback entries with HTF bias
            # Long: price>12h_HMA + pullback to EMA21 + RSI>40 (not oversold)
            # Short: price<12h_HMA + rally to EMA21 + RSI<60 (not overbought)
            
            pullback_to_ema_long = close[i] <= ema_21[i] * 1.005  # within 0.5% of EMA
            pullback_to_ema_short = close[i] >= ema_21[i] * 0.995  # within 0.5% of EMA
            
            if price_above_hma_12h and pullback_to_ema_long and rsi_14[i] > 40:
                # Long pullback in bullish trend
                desired_signal = BASE_SIZE
            
            elif price_below_hma_12h and pullback_to_ema_short and rsi_14[i] < 60:
                # Short rally in bearish trend
                desired_signal = -BASE_SIZE
        
        elif is_ranging:
            # RANGE REGIME: Bollinger Band mean reversion with HTF bias
            # Long: price<BB_lower + RSI<35 + price>12h_HMA (bullish macro)
            # Short: price>BB_upper + RSI>65 + price<12h_HMA (bearish macro)
            
            at_bb_lower = close[i] <= bb_lower[i]
            at_bb_upper = close[i] >= bb_upper[i]
            
            if price_above_hma_12h and at_bb_lower and rsi_14[i] < 35:
                # Long at BB lower in bullish macro (range)
                desired_signal = BASE_SIZE
            
            elif price_below_hma_12h and at_bb_upper and rsi_14[i] > 65:
                # Short at BB upper in bearish macro (range)
                desired_signal = -BASE_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position in hysteresis zone or if regime still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if: in hysteresis OR trend regime with bullish bias
                if (20.0 <= adx_value <= 25.0) or \
                   (is_trending and price_above_hma_12h) or \
                   (is_ranging and price_above_hma_12h and rsi_14[i] < 60):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if: in hysteresis OR trend regime with bearish bias
                if (20.0 <= adx_value <= 25.0) or \
                   (is_trending and price_below_hma_12h) or \
                   (is_ranging and price_below_hma_12h and rsi_14[i] > 40):
                    desired_signal = -BASE_SIZE
        
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