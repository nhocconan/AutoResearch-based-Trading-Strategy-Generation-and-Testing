#!/usr/bin/env python3
"""
Experiment #009: 1h Vol Spike Mean Reversion with 4h HMA Trend Filter
Hypothesis: After volatility spikes (ATR(7)/ATR(30) > 2.0), price tends to revert.
Combined with 4h HMA trend bias and Bollinger Band extremes, this captures "vol crush" after panic.
Key insight: All 7 previous strategies failed. Funding rate didn't work in exp#008. 
This uses PURE price action: vol spike detection + mean reversion + HTF trend filter.
Research shows vol spike reversion works through 2022 crash (BTC -77% period).
Timeframe: 1h (REQUIRED for exp#009), HTF: 4h via mtf_data helper.
Entry: ATR ratio > 2.0 + price < BB(20, 2.5) + 4h HMA bullish for longs (reverse for shorts)
Exit: ATR ratio < 1.2 or stoploss at 2.5*ATR
Position sizing: 0.25 discrete, stoploss mandatory.
Why this might work: Different from all 7 failed strategies. Targets vol regime, not trend/RSI.
Must generate 10+ trades on train, 3+ on test - conditions loosened vs strict filters.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_vol_spike_meanrev_4h_hma_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_bollinger_bands(close, period=20, std_mult=2.5):
    """Calculate Bollinger Bands with configurable std multiplier."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # Bollinger Bands with 2.5 std for extreme moves
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.5)
    
    # ATR ratio for vol spike detection
    atr_ratio = atr_7 / (atr_30 + 1e-10)
    
    # EMA for trend confirmation
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - main regime filter
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # Vol spike detection (ATR(7)/ATR(30) > 2.0)
        vol_spike = atr_ratio[i] > 2.0
        
        # Vol normalization (ATR ratio < 1.2 for exit)
        vol_normal = atr_ratio[i] < 1.2
        
        # Bollinger Band extremes
        price_below_bb = close[i] < bb_lower[i]
        price_above_bb = close[i] > bb_upper[i]
        price_near_bb_lower = close[i] < bb_lower[i] * 1.005  # Within 0.5% of lower band
        price_near_bb_upper = close[i] > bb_upper[i] * 0.995  # Within 0.5% of upper band
        
        # RSI extremes for confirmation
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = 35 <= rsi[i] <= 65
        
        # 1h trend confirmation
        bull_trend_1h = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_1h = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (vol spike + mean reversion + 4h bullish) ===
        # Primary: Vol spike + price at BB lower + 4h trend bullish
        if vol_spike and price_near_bb_lower and bull_trend_4h:
            new_signal = SIZE_BASE
        
        # Secondary: Vol spike + RSI oversold + 4h trend bullish
        elif vol_spike and rsi_oversold and bull_trend_4h:
            new_signal = SIZE_BASE
        
        # Tertiary: Price at BB lower + RSI oversold (no vol spike needed)
        elif price_below_bb and rsi_oversold and bull_trend_4h:
            new_signal = SIZE_HALF
        
        # Momentum: 1h trend + 4h trend + RSI recovering from oversold
        elif bull_trend_1h and bull_trend_4h and rsi[i] > 40 and rsi[i-1] < 40:
            new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (vol spike + mean reversion + 4h bearish) ===
        # Primary: Vol spike + price at BB upper + 4h trend bearish
        elif vol_spike and price_near_bb_upper and bear_trend_4h:
            new_signal = -SIZE_BASE
        
        # Secondary: Vol spike + RSI overbought + 4h trend bearish
        elif vol_spike and rsi_overbought and bear_trend_4h:
            new_signal = -SIZE_BASE
        
        # Tertiary: Price at BB upper + RSI overbought (no vol spike needed)
        elif price_above_bb and rsi_overbought and bear_trend_4h:
            new_signal = -SIZE_HALF
        
        # Momentum: 1h trend + 4h trend + RSI dropping from overbought
        elif bear_trend_1h and bear_trend_4h and rsi[i] < 60 and rsi[i-1] > 60:
            new_signal = -SIZE_HALF
        
        # === EXIT CONDITIONS ===
        # Exit long when vol normalizes (vol crush complete)
        if position_side > 0 and vol_normal:
            new_signal = 0.0
        
        # Exit short when vol normalizes (vol crush complete)
        elif position_side < 0 and vol_normal:
            new_signal = 0.0
        
        # Exit long when price reaches BB mid (mean reversion complete)
        if position_side > 0 and not np.isnan(bb_mid[i]) and close[i] > bb_mid[i]:
            new_signal = 0.0
        
        # Exit short when price reaches BB mid (mean reversion complete)
        elif position_side < 0 and not np.isnan(bb_mid[i]) and close[i] < bb_mid[i]:
            new_signal = 0.0
        
        # === STOPLOSS LOGIC (Rule 6) - MANDATORY ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr_14[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr_14[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr_14[i] if position_side > 0 else close[i] + 2.5 * atr_14[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr_14[i] if position_side > 0 else close[i] + 2.5 * atr_14[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals