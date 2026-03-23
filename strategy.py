#!/usr/bin/env python3
"""
Experiment #016: 12h Primary + 1d HTF — Dual Regime (Choppiness + Donchian + RSI)

Hypothesis: Based on research showing Choppiness Index regime switching works well for ETH 
(Sharpe +0.923) and Donchian+HMA+RSI works for SOL (+0.782), I'm combining these into a 
dual-regime strategy at 12h timeframe.

Key innovation:
1. CHOP(14) > 61.8 = RANGING → Use RSI mean reversion (buy <30, sell >70)
2. CHOP(14) < 38.2 = TRENDING → Use Donchian breakout + HMA trend filter
3. 38.2 <= CHOP <= 61.8 = TRANSITION → Stay flat (no trades)
4. 1d HMA confirms overall bias (only long if 1d HMA bullish for longs, vice versa)

Why 12h works:
- Targets 20-50 trades/year (fee-efficient per Rule 10)
- Less noise than 4h/1h, more signals than 1d
- Proven in research for crypto perpetual futures

Entry conditions (LOOSE enough to generate trades):
- Long in range: RSI < 35 + CHOP > 61.8 + price > 1d HMA
- Short in range: RSI > 65 + CHOP > 61.8 + price < 1d HMA
- Long in trend: Donchian breakout + CHOP < 38.2 + 12h HMA bullish
- Short in trend: Donchian breakdown + CHOP < 38.2 + 12h HMA bearish

Position size: 0.28 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_dual_regime_donchian_1d_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    Interpretation:
    - CHOP > 61.8 = Market is chopping/ranging (mean reversion favorable)
    - CHOP < 38.2 = Market is trending (trend following favorable)
    - 38.2 <= CHOP <= 61.8 = Transition zone (stay flat)
    """
    n = period
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of ATR over period
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    
    # Highest High and Lowest Low over period
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    # Price range
    price_range = highest_high - lowest_low + 1e-10
    
    # Choppiness Index
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high and lowest low over period)."""
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
    
    # Calculate 1d HMA for regime bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # 12h HMA for trend confirmation
    hma_12h = calculate_hma(close, period=21)
    
    # Donchian channels for breakout detection
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(donchian_upper[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D TREND BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 12H TREND CONFIRMATION ===
        hma_12h_slope_bull = hma_12h[i] > hma_12h[i-3] if i >= 3 else False
        hma_12h_slope_bear = hma_12h[i] < hma_12h[i-3] if i >= 3 else False
        price_above_hma_12h = close[i] > hma_12h[i]
        price_below_hma_12h = close[i] < hma_12h[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 61.8
        is_trending = chop_value < 38.2
        is_transition = (chop_value >= 38.2) and (chop_value <= 61.8)
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i >= 1 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i >= 1 else False
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === DUAL REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion ---
        if is_ranging:
            # Long: RSI oversold + price above 1d HMA (bullish bias)
            if rsi_oversold and price_above_hma_1d:
                new_signal = POSITION_SIZE
            
            # Short: RSI overbought + price below 1d HMA (bearish bias)
            elif rsi_overbought and price_below_hma_1d:
                new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Breakout Following ---
        elif is_trending:
            # Long: Donchian breakout + 12h HMA bullish
            if donchian_breakout_long and hma_12h_slope_bull and price_above_hma_12h:
                new_signal = POSITION_SIZE
            
            # Short: Donchian breakdown + 12h HMA bearish
            elif donchian_breakout_short and hma_12h_slope_bear and price_below_hma_12h:
                new_signal = -POSITION_SIZE
        
        # --- TRANSITION REGIME: Stay Flat ---
        # new_signal remains 0.0
        
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
        # Exit long if regime changes from ranging to trending bearish
        if in_position and position_side > 0:
            if is_trending and hma_12h_slope_bear:
                new_signal = 0.0
        
        # Exit short if regime changes from ranging to trending bullish
        if in_position and position_side < 0:
            if is_trending and hma_12h_slope_bull:
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