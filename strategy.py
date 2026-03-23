#!/usr/bin/env python3
"""
Experiment #065: 1h Primary + 4h/1d HTF — Simplified Regime-Adaptive with Relaxed Entries

Hypothesis: Previous 1h strategies failed with 0 trades due to overly strict confluence.
This version uses RELAXED entry thresholds to ensure 40-80 trades/year while maintaining
HTF trend bias for direction. Key changes from failed #055/#058/#060:
1) RSI thresholds widened: 35/65 instead of 15/85 (CRSI was too rare)
2) Removed session filter (was eliminating 60% of bars)
3) Removed volume spike requirement (too strict for 1h)
4) 4h HMA is soft bias, not hard filter
5) Added ADX hysteresis: trend mode >25, range mode <18
6) Fallback mean-reversion in low-ADX regimes

Why this should work:
- 1h timeframe with 4h trend bias (proven in #061)
- Relaxed entries ensure trade generation (critical - 0 trades = reject)
- Regime-adaptive: trend follow when ADX>25, mean revert when ADX<18
- ATR stoploss prevents catastrophic drawdown
- Position size 0.25-0.30 (conservative for 1h)

Target: 40-80 trades/year, Sharpe > 0.5, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_adx_regime_4h1d_relaxed_v1"
timeframe = "1h"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx.fillna(0.0).values
    return adx

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28  # Discrete: 0.28 (between 0.25-0.30)
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === HTF TREND BIAS (soft filter, not hard) ===
        bullish_4h = close[i] > hma_4h_aligned[i]
        bearish_4h = close[i] < hma_4h_aligned[i]
        bullish_1d = close[i] > hma_1d_aligned[i]
        bearish_1d = close[i] < hma_1d_aligned[i]
        
        # Strong bias when 4h and 1d agree
        strong_bullish = bullish_4h and bullish_1d
        strong_bearish = bearish_4h and bearish_1d
        
        # === ADX REGIME DETECTION ===
        adx_value = adx_14[i]
        is_trending = adx_value > 25.0  # Trending market
        is_ranging = adx_value < 18.0   # Ranging market
        # ADX 18-25 = transition zone (use both strategies)
        
        # === RSI SIGNALS (relaxed thresholds for trade generation) ===
        rsi_value = rsi_14[i]
        rsi_oversold = rsi_value < 40.0   # Was 35, relaxed for more trades
        rsi_overbought = rsi_value > 60.0  # Was 65, relaxed for more trades
        
        # === BOLLINGER BAND SIGNALS ===
        at_bb_lower = close[i] <= bb_lower[i] * 1.002  # Near or below lower band
        at_bb_upper = close[i] >= bb_upper[i] * 0.998  # Near or above upper band
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        # --- TRENDING REGIME: Follow HTF trend on RSI pullback ---
        if is_trending:
            # Long: HTF bullish + RSI pullback (not extreme oversold)
            if bullish_4h and rsi_value < 55.0 and rsi_value > 35.0:
                new_signal = POSITION_SIZE
            
            # Short: HTF bearish + RSI pullback (not extreme overbought)
            elif bearish_4h and rsi_value > 45.0 and rsi_value < 65.0:
                new_signal = -POSITION_SIZE
        
        # --- RANGING REGIME: Mean reversion at BB extremes ---
        elif is_ranging:
            # Long: At BB lower + RSI oversold
            if at_bb_lower and rsi_oversold:
                # Only if not strongly bearish on 1d
                if not strong_bearish:
                    new_signal = POSITION_SIZE
            
            # Short: At BB upper + RSI overbought
            elif at_bb_upper and rsi_overbought:
                # Only if not strongly bullish on 1d
                if not strong_bullish:
                    new_signal = -POSITION_SIZE
        
        # --- TRANSITION ZONE (ADX 18-25): Hybrid approach ---
        else:
            # Long: HTF bullish + RSI oversold
            if bullish_4h and rsi_oversold:
                new_signal = POSITION_SIZE
            
            # Short: HTF bearish + RSI overbought
            elif bearish_4h and rsi_overbought:
                new_signal = -POSITION_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON HTF TREND REVERSAL ===
        if in_position and position_side > 0:
            # Exit long if 4h and 1d both turn bearish
            if strong_bearish:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h and 1d both turn bullish
            if strong_bullish:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals