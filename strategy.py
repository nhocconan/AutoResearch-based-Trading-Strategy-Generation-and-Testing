#!/usr/bin/env python3
"""
Experiment #041: 4h Primary + 1d HTF — Volatility Spike Mean Reversion + Fisher Transform

Hypothesis: Volatility spikes (ATR ratio > 1.8) indicate panic/capitulation which often reverses.
Combined with Fisher Transform for precise turning point detection and 1d HMA for macro bias,
this should generate 30-50 trades/year with Sharpe > 0.486.

Why this should work (different from 39 failed experiments):
1) Fisher Transform catches reversals better than RSI/CRSI (proven in literature)
2) Vol spike detection = high-probability mean reversion setups
3) Looser thresholds (ATR ratio 1.8 not 2.0, Fisher -1.5/+1.5) = ensures trades
4) 1d HMA as soft filter (not hard requirement) = avoids over-filtering
5) Simple regime logic = less chance of 0 trades

Position size: 0.30 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
Target: 30-50 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_volspike_regime_1d_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(close)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        # Normalize price to -1 to +1 range
        range_val = highest - lowest
        if range_val < 1e-10:
            range_val = 1e-10
        
        normalized = (hl2 - lowest) / range_val
        
        # Apply Fisher transform formula
        fisher_raw = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
        
        # Smooth with EMA
        if i == period:
            fisher[i] = fisher_raw
            trigger[i] = fisher_raw
        else:
            fisher[i] = 0.7 * fisher_raw + 0.3 * fisher[i-1]
            trigger[i] = fisher[i-1]
    
    return fisher, trigger

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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    rsi_14 = calculate_rsi(close, period=14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, period=9)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Volatility spike ratio (ATR short / ATR long)
    vol_ratio = atr_14 / (atr_30 + 1e-10)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30  # Discrete, within 0.20-0.35 range
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):  # Warmup for all indicators
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(bb_upper[i]) or np.isnan(rsi_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D MACRO BIAS (soft filter, not hard requirement) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        daily_bullish = price_above_hma_1d
        daily_bearish = price_below_hma_1d
        
        # === VOLATILITY SPIKE DETECTION (loose threshold for trades) ===
        vol_spike = vol_ratio[i] > 1.8  # ATR(14) > 1.8 * ATR(30)
        vol_extreme = vol_ratio[i] > 2.2  # Very extreme
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = fisher[i] > -1.5 and fisher_trigger[i] <= -1.5
        fisher_cross_down = fisher[i] < 1.5 and fisher_trigger[i] >= 1.5
        fisher_rising = fisher[i] > fisher[i-1] if i > 0 else False
        fisher_falling = fisher[i] < fisher[i-1] if i > 0 else False
        
        # === RSI EXTREMES (backup filter, loose) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_oversold = rsi_14[i] < 25.0
        rsi_extreme_overbought = rsi_14[i] > 75.0
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        bb_squeeze = (bb_upper[i] - bb_lower[i]) / bb_mid[i] < 0.05
        
        # === ENTRY LOGIC - DESIGNED TO GENERATE TRADES ===
        new_signal = 0.0
        
        # --- LONG ENTRIES (vol spike + reversal) ---
        # Primary: Vol spike + Fisher oversold + daily helps
        if vol_spike and fisher_oversold:
            if daily_bullish or rsi_oversold or price_below_bb_lower:
                new_signal = POSITION_SIZE
        
        # Secondary: Fisher cross up + RSI oversold (no vol spike needed)
        elif fisher_cross_up and rsi_oversold:
            if daily_bullish or vol_ratio[i] > 1.5:
                new_signal = POSITION_SIZE
        
        # Tertiary: BB break + RSI extreme (mean reversion)
        elif price_below_bb_lower and rsi_extreme_oversold:
            if fisher_rising or daily_bullish:
                new_signal = POSITION_SIZE
        
        # --- SHORT ENTRIES (vol spike + reversal) ---
        # Primary: Vol spike + Fisher overbought + daily helps
        if new_signal == 0.0 and vol_spike and fisher_overbought:
            if daily_bearish or rsi_overbought or price_above_bb_upper:
                new_signal = -POSITION_SIZE
        
        # Secondary: Fisher cross down + RSI overbought
        elif new_signal == 0.0 and fisher_cross_down and rsi_overbought:
            if daily_bearish or vol_ratio[i] > 1.5:
                new_signal = -POSITION_SIZE
        
        # Tertiary: BB break + RSI extreme
        elif new_signal == 0.0 and price_above_bb_upper and rsi_extreme_overbought:
            if fisher_falling or daily_bearish:
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
        
        # === EXIT ON FISHER REVERSAL (take profit) ===
        if in_position and position_side > 0:
            if fisher[i] > 1.5:  # Fisher overbought = exit long
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if fisher[i] < -1.5:  # Fisher oversold = exit short
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