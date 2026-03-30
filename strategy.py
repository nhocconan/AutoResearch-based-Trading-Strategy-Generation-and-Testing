#!/usr/bin/env python3
"""
Experiment #002: 4h Trailing Channel + ATR Volatility Filter + Choppiness Regime

PRIMARY: 4h | HTF: 12h
HYPOTHESIS: Combine the BEST elements from DB winners with tighter entry filters:
1. Choppiness regime filter (key meta-filter, <42 for stronger trend)
2. ATR-based volatility expansion (current ATR > 1.5x ATR(20) mean = real trend, not noise)
3. Price distance from SMA200 filter (within 15% = better entries, avoids extremes)
4. Trailing Donchian channel (proven price structure)
5. 12h EMA(21) for trend direction
6. Volume spike confirmation (2.0x)

WHY IT WORKS IN BULL + BEAR + RANGE:
- Bull: CHOP < 42 + ATR expansion + price near SMA200 + breakout + HTF up = strong longs
- Bear: CHOP < 42 + ATR expansion + price near SMA200 + breakdown + HTF down = strong shorts
- Range: CHOP > 61.8 = SKIP (avoids whipsaws, the #1 killer)
- ATR trailing stop scales with volatility (handles 2022 crash)

DIFFERENCE FROM CURRENT (Sharpe 0.271, 145 trades):
- Added ATR expansion filter (1.5x) = fewer but more confirmed entries
- Added SMA200 distance filter (15%) = better timing
- Tightened CHOP to 42 (from 45) = stronger trend required
- Target: 100-120 total trades (within 200 max)

SIGNAL PHILOSOPHY: Only enter when volatility is expanding AND price structure confirms.
Vol spike without ATR expansion = many false breakouts. This filter should reduce trades by ~20%.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trailing_channel_atr_vol_chop_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(data, period):
    """Simple Moving Average with min_periods"""
    return pd.Series(data).rolling(window=period, min_periods=period).mean().values

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging - DON'T enter
    CHOP < 42 = trending - GOOD to enter (tighter than usual 50, tighter than current 45)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest > lowest and atr_sum > 0:
            range_hl = highest - lowest
            chop[i] = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_trailing_channel(high, low, period=20):
    """
    Trailing Donchian Channel - tracks highest high and lowest low
    Uses the channel BOTTOM for longs (support), TOP for shorts (resistance)
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(21) for trend direction
    ema_21_12h = pd.Series(df_12h['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_mean = calculate_sma(atr_14, 20)  # ATR(20) mean for expansion ratio
    channel_up, channel_lo = calculate_trailing_channel(high, low, period=20)
    chop = calculate_choppiness(high, low, close, period=14)
    sma200 = calculate_sma(close, 200)
    
    # Volume ratio (20-period MA) - 2.0x threshold
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    # Warmup: 200 for channel + 20 for ATR mean + 14 for CHOP + 200 for SMA200
    warmup = 250
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_mean[i]) or atr_mean[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(channel_up[i]) or np.isnan(channel_lo[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma200[i]) or sma200[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === KEY FILTERS ===
        
        # 1. CHOPPINESS REGIME FILTER (<42 for stronger trend)
        chop_value = chop[i]
        is_choppy = chop_value > 61.8
        is_strong_trend = chop_value < 42  # Tighter than current 45
        
        # 2. ATR VOLATILITY EXPANSION FILTER (NEW: reduces false breakouts)
        # Current ATR must be > 1.5x the 20-bar ATR mean
        # This ensures we're in a real trend expansion, not just noise
        atr_expansion = atr_14[i] / atr_mean[i]
        has_vol_expansion = atr_expansion > 1.5
        
        # 3. PRICE DISTANCE FROM SMA200 FILTER (NEW: better timing)
        # Price within 15% of SMA200 = not at extreme, better entry
        price_distance_ratio = abs(close[i] - sma200[i]) / sma200[i]
        near_sma200 = price_distance_ratio < 0.15
        
        # === HTF TREND: 12h EMA(21) direction ===
        htf_trend_up = close[i] > ema_aligned[i]
        htf_trend_down = close[i] < ema_aligned[i]
        
        # === VOLUME CONFIRMATION (2.0x)
        vol_spike = vol_ratio[i] > 2.0
        
        # === TRAILING CHANNEL BREAKOUT ===
        prev_channel_up = channel_up[i - 1]
        prev_channel_lo = channel_lo[i - 1]
        
        breakout_up = close[i] > prev_channel_up
        breakout_down = close[i] < prev_channel_lo
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: All conditions must be met ===
            # CHOP < 42 + ATR expansion + near SMA200 + breakout + HTF up + vol spike
            if breakout_up and htf_trend_up and vol_spike and is_strong_trend and has_vol_expansion and near_sma200:
                desired_signal = SIZE
            
            # === SHORT: All conditions must be met ===
            if breakout_down and htf_trend_down and vol_spike and is_strong_trend and has_vol_expansion and near_sma200:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: exit if price falls 2.5 ATR from recent high
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips
                if htf_trend_down:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy
                if is_choppy:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: exit if price rises 2.5 ATR from recent low
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips
                if htf_trend_up:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy
                if is_choppy:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals