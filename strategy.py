#!/usr/bin/env python3
"""
Experiment #019: 15m Donchian Breakout with 4h HMA Trend + Volume Confirmation

Hypothesis: After 18 failed experiments, the pattern shows:
1. Complex mean-reversion + trend hybrids fail on lower TFs (15m/30m)
2. RSI/ADX filters create too many conflicting conditions = few trades
3. Simple breakout strategies with HTF trend filter may work better

This 15m strategy uses:

1. Donchian Channel(20): Classic breakout system. Long when price breaks 
   20-bar high, short when breaks 20-bar low. Proven in crypto momentum.

2. 4h HMA(21) Trend Filter: Only take long breakouts if price > 4h_HMA,
   only take short breakouts if price < 4h_HMA. Prevents counter-trend trades.

3. Volume Confirmation: Breakout volume must be > 1.5x 20-bar avg volume.
   Filters false breakouts (major cause of 15m strategy failures).

4. ATR(14) Stoploss: 2.5*ATR trailing stop to protect from reversals.
   Critical for 15m where noise causes quick whipsaws.

5. Discrete Position Sizing: 0.25 base size, reduced to 0.15 in low-volume.
   Minimizes fee churn while maintaining exposure.

Why this should beat #007/#013 (both 15m failures):
- Simpler logic = fewer conflicting filters = MORE trades (critical!)
- Volume confirmation replaces ADX (ADX failed on 15m twice)
- Donchian breakout works in both bull AND bear markets
- 4h HMA more stable than 1h for trend filter on 15m entries
- Target 60-100 trades/year (optimal for 15m per Rule 10)

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.15-0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_donchian_4h_hma_vol_confirm_atr_v1"
timeframe = "15m"
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

def calculate_donchian_channels(high, low, period=20):
    """
    Calculate Donchian Channel upper and lower bounds.
    Upper = highest high over period
    Lower = lowest low over period
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_volume_ma(volume, period=20):
    """Calculate rolling moving average of volume."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    volume_ma_20 = calculate_volume_ma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25      # Normal position
    REDUCED_SIZE = 0.15   # Low volume confirmation
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(volume_ma_20[i]) or volume_ma_20[i] == 0:
            continue
        
        # === 4H HMA TREND BIAS (HTF filter) ===
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Long: price breaks above 20-bar high
        breakout_long = close[i] > donchian_upper[i-1]  # Use previous bar's upper
        
        # Short: price breaks below 20-bar low
        breakout_short = close[i] < donchian_lower[i-1]  # Use previous bar's lower
        
        # === VOLUME CONFIRMATION ===
        # Volume must be > 1.5x average for confirmed breakout
        vol_ratio = volume[i] / volume_ma_20[i]
        high_volume = vol_ratio > 1.5
        normal_volume = vol_ratio > 1.0
        
        # === POSITION SIZING BASED ON VOLUME ===
        if high_volume:
            position_size = BASE_SIZE
        elif normal_volume:
            position_size = REDUCED_SIZE
        else:
            position_size = 0.0  # Skip low volume breakouts
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long entry: Donchian breakout + bull trend + volume confirmation
        if breakout_long and bull_trend and position_size > 0:
            new_signal = position_size
        
        # Short entry: Donchian breakout + bear trend + volume confirmation
        elif breakout_short and bear_trend and position_size > 0:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_exit = False
        if in_position and position_side != 0:
            # Exit long if trend turns bearish
            if position_side > 0 and bear_trend:
                trend_exit = True
            # Exit short if trend turns bullish
            if position_side < 0 and bull_trend:
                trend_exit = True
        
        # Apply stoploss or trend exit
        if stoploss_triggered or trend_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
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
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals