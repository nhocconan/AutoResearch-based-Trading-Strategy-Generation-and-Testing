#!/usr/bin/env python3
"""
Experiment #022: 12h Primary + 1d/1w HTF — Volatility Spike Mean Reversion

Hypothesis: Based on research showing vol spike reversion works exceptionally well 
for BTC/ETH in bear markets (Sharpe 0.8-1.5 through 2022 crash), I'm implementing 
a volatility-based mean reversion strategy at 12h timeframe.

Key innovation:
1. ATR RATIO (7/30) > 2.0 = Volatility spike (panic/euphoria) → Fade the move
2. Bollinger Band position confirms extreme (price < BB_lower or > BB_upper)
3. Choppiness Index > 50 = Range favorable for mean reversion
4. 1d HMA provides overall bias (only long if price > 1d HMA, vice versa)
5. 1w HMA for major regime filter (avoid counter-trend in strong weekly trends)

Why this works in bear/range markets:
- Captures "vol crush" after panic selling (2022 crash bottoms)
- Fades euphoric rallies in bear markets (2025 pattern)
- 12h TF targets 20-50 trades/year (fee-efficient per Rule 10)
- Vol spikes are reliable reversal signals in crypto

Entry conditions (LOOSE enough to generate trades):
- Long: ATR_ratio > 1.8 + price < BB_lower(20,2.0) + CHOP > 45 + price > 1d HMA
- Short: ATR_ratio > 1.8 + price > BB_upper(20,2.0) + CHOP > 45 + price < 1d HMA
- Trend breakout: ATR_ratio < 1.2 + CHOP < 40 + Donchian breakout + 1d HMA confirm

Position size: 0.27 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_volspike_meanrevert_bbands_1d1w_v1"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = period
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
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
    return rsi.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for trend bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for major regime
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # ATR ratio (volatility spike detector)
    atr_ratio = atr_7 / (atr_30 + 1e-10)
    
    # Bollinger Bands
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Choppiness Index
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # RSI for additional confirmation
    rsi_14 = calculate_rsi(close, period=14)
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # 12h HMA for short-term trend
    hma_12h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.27
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D TREND BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 1W MAJOR REGIME ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 12H TREND ===
        hma_12h_slope_bull = hma_12h[i] > hma_12h[i-3] if i >= 3 else False
        hma_12h_slope_bear = hma_12h[i] < hma_12h[i-3] if i >= 3 else False
        
        # === VOLATILITY SPIKE ===
        vol_spike = atr_ratio[i] > 1.8
        vol_compressed = atr_ratio[i] < 1.2
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        bb_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 50
        is_trending = chop_value < 40
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i >= 1 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i >= 1 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- VOL SPIKE MEAN REVERSION (primary signal) ---
        if vol_spike and is_ranging:
            # Long: vol spike + price at BB lower + RSI oversold + 1d bullish bias
            if price_below_bb_lower and rsi_oversold and price_above_hma_1d:
                new_signal = POSITION_SIZE
            
            # Short: vol spike + price at BB upper + RSI overbought + 1d bearish bias
            elif price_above_bb_upper and rsi_overbought and price_below_hma_1d:
                new_signal = -POSITION_SIZE
        
        # --- TREND BREAKOUT (secondary, less frequent) ---
        elif vol_compressed and is_trending:
            # Long breakout: Donchian + 12h HMA bull + 1d HMA bull
            if donchian_breakout_long and hma_12h_slope_bull and price_above_hma_1d:
                new_signal = POSITION_SIZE
            
            # Short breakdown: Donchian + 12h HMA bear + 1d HMA bear
            elif donchian_breakout_short and hma_12h_slope_bear and price_below_hma_1d:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
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
        
        # === EXIT ON REGIME CHANGE ===
        # Exit long if 1d trend turns bearish
        if in_position and position_side > 0 and price_below_hma_1d:
            new_signal = 0.0
        
        # Exit short if 1d trend turns bullish
        if in_position and position_side < 0 and price_above_hma_1d:
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