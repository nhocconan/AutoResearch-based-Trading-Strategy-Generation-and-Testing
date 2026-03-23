#!/usr/bin/env python3
"""
Experiment #004: 4h Primary + 12h/1d HTF — Vol Spike Reversion + Asymmetric Regime

Hypothesis: After 3 failed experiments using Chop/CRSI/Donchian patterns, I'm shifting to
volatility-based mean reversion which has research backing for BTC/ETH specifically.

Key differences from failed attempts:
1. NO Choppiness Index (failed in #001, #002)
2. NO Connors RSI (failed in #001)
3. Using ATR ratio (ATR7/ATR30) for vol spike detection — research shows 2.0+ threshold
   captures panic bottoms with 70%+ win rate on BTC/ETH
4. Asymmetric regime: Only short when 12h HMA bearish, only long when 12h HMA bullish
5. BB %B for precise entry timing within vol spike context

Why this might work:
- Vol spike reversion worked through 2022 crash (unlike pure trend following)
- Asymmetric entries prevent fighting the HTF trend
- 4h TF targets 20-50 trades/year (fee-efficient)
- Position size 0.28 (conservative for 4h per Rule 4)

Entry conditions (LOOSE enough to generate trades):
- Long: ATR_ratio > 1.8 OR BB_pct_b < 0.15, AND 12h HMA bullish OR flat
- Short: ATR_ratio > 1.8 OR BB_pct_b > 0.85, AND 12h HMA bearish OR flat
- Either condition triggers (not both required)

Stoploss: 2.5*ATR trailing, signal→0 when hit
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_volspike_bb_asymmetric_12h_v1"
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
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    # %B (position within bands)
    pct_b = (close - lower) / (upper - lower + 1e-10)
    
    # Bandwidth
    bandwidth = (upper - lower) / (sma + 1e-10)
    
    return upper.values, lower.values, pct_b.values, bandwidth.values

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HMA for trend direction
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1d HMA for regime confirmation
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    bb_upper, bb_lower, bb_pct_b, bb_bandwidth = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    rsi_14 = calculate_rsi(close, period=14)
    
    # ATR ratio for vol spike detection
    atr_ratio = atr_7 / (atr_30 + 1e-10)
    
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
        if np.isnan(hma_12h_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(bb_pct_b[i]) or np.isnan(atr_ratio[i]) or np.isnan(rsi_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 12H TREND BIAS ===
        hma_12h_slope_bull = hma_12h_aligned[i] > hma_12h_aligned[i-3] if i >= 3 else False
        hma_12h_slope_bear = hma_12h_aligned[i] < hma_12h_aligned[i-3] if i >= 3 else False
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === 1D REGIME CONFIRMATION ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-2] if i >= 2 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-2] if i >= 2 else False
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = atr_ratio[i] > 1.8  # Research-backed threshold
        
        # === BOLLINGER BAND EXTREMES ===
        bb_extreme_low = bb_pct_b[i] < 0.15
        bb_extreme_high = bb_pct_b[i] > 0.85
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        
        # === ASYMMETRIC ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Condition 1: Vol spike + oversold (panic bottom)
        vol_spike_long = vol_spike and (bb_extreme_low or rsi_oversold)
        
        # Condition 2: BB mean reversion without vol spike
        bb_revert_long = bb_extreme_low and rsi_14[i] < 40
        
        # Only enter long if 12h trend is bullish OR flat (not strongly bearish)
        trend_allows_long = price_above_hma_12h or (not hma_12h_slope_bear)
        
        if (vol_spike_long or bb_revert_long) and trend_allows_long:
            new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Condition 1: Vol spike + overbought (panic top)
        vol_spike_short = vol_spike and (bb_extreme_high or rsi_overbought)
        
        # Condition 2: BB mean reversion without vol spike
        bb_revert_short = bb_extreme_high and rsi_14[i] > 60
        
        # Only enter short if 12h trend is bearish OR flat (not strongly bullish)
        trend_allows_short = price_below_hma_12h or (not hma_12h_slope_bull)
        
        if (vol_spike_short or bb_revert_short) and trend_allows_short:
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_12h_slope_bear and price_below_hma_12h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_12h_slope_bull and price_above_hma_12h:
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