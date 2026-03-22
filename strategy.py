#!/usr/bin/env python3
"""
Experiment #001: 4h HMA-Donchian with 1d Trend Filter + Choppiness Regime

Hypothesis: 4h timeframe captures major moves while filtering noise. Combining:
1. 4h HMA crossover for trend direction (fast 16 vs slow 48)
2. 4h Donchian(20) breakout for entry timing
3. 1d HMA(21) for major trend bias (via mtf_data helper)
4. Choppiness Index(14) for regime detection (trend vs mean-revert)
5. RSI(14) filter to avoid extreme entries
6. ATR(14) trailing stoploss at 2.5x

Why 4h works:
- Natural 20-50 trades/year (fee drag ~1-2.5%)
- Filters 15m/1h noise but catches moves before 12h/1d
- Proven in prior experiments (SOL Sharpe +0.782, ETH +0.755)

Regime logic:
- CHOP > 61.8: Range market → mean revert at Donchian bounds
- CHOP < 38.2: Trend market → breakout entries only
- Between: Use trend following with stricter filters

Timeframe: 4h (REQUIRED for Experiment #001)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_donchian_chop_regime_1d_filter_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    atr = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0, posinf=50.0, neginf=50.0)
    chop = np.clip(chop, 0, 100)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_4h_16 = calculate_hma(close, 16)
    hma_4h_48 = calculate_hma(close, 48)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_signal = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = last_signal
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            signals[i] = last_signal
            continue
        
        if np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            signals[i] = last_signal
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = last_signal
            continue
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H HMA TREND ===
        hma_bullish = hma_4h_16[i] > hma_4h_48[i]
        hma_bearish = hma_4h_16[i] < hma_4h_48[i]
        
        # === HMA SLOPE CONFIRMATION ===
        hma_slope_long = hma_4h_16[i] > hma_4h_16[i-1] if i > 0 else False
        hma_slope_short = hma_4h_16[i] < hma_4h_16[i-1] if i > 0 else False
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI FILTER ===
        rsi_ok_long = rsi_14[i] < 70
        rsi_ok_short = rsi_14[i] > 30
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_trending = chop_value < 45.0  # Below 45 = trending regime
        is_ranging = chop_value > 55.0  # Above 55 = ranging regime
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY
        if is_trending:
            # Trend regime: breakout + trend alignment
            if hma_bullish and breakout_long and rsi_ok_long:
                if daily_bullish:
                    new_signal = current_size
                elif not daily_bearish:
                    new_signal = current_size * 0.8
        elif is_ranging:
            # Range regime: mean revert at lower Donchian
            if close[i] < donchian_lower[i-1] * 1.02 and rsi_14[i] < 40:
                new_signal = current_size * 0.6
        else:
            # Neutral regime: strict trend following
            if hma_bullish and hma_slope_long and breakout_long and daily_bullish:
                new_signal = current_size
        
        # SHORT ENTRY
        if is_trending:
            # Trend regime: breakout + trend alignment
            if hma_bearish and breakout_short and rsi_ok_short:
                if daily_bearish:
                    new_signal = -current_size
                elif not daily_bullish:
                    new_signal = -current_size * 0.8
        elif is_ranging:
            # Range regime: mean revert at upper Donchian
            if close[i] > donchian_upper[i-1] * 0.98 and rsi_14[i] > 60:
                new_signal = -current_size * 0.6
        else:
            # Neutral regime: strict trend following
            if hma_bearish and hma_slope_short and breakout_short and daily_bearish:
                new_signal = -current_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if position_side > 0 and hma_bearish:
            new_signal = 0.0
        if position_side < 0 and hma_bullish:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if position_side == 0:
                # New entry
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if position_side != 0:
                # Exit position
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
        last_signal = new_signal
    
    return signals