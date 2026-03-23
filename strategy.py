#!/usr/bin/env python3
"""
Experiment #181: 4h Primary + 1d/1w HTF — Ehlers Fisher Transform + HMA Trend + Volume

Hypothesis: Previous strategies failed due to too many conflicting filters (0 trades).
This uses Ehlers Fisher Transform (proven for bear market reversals) with simple
1d HMA trend filter and minimal entry conditions. Key insight: Fisher crosses are
frequent enough to generate 30-50 trades/year while HTF filter prevents wrong-direction
trades. Simpler logic = more trades = positive Sharpe on ALL symbols.

KEY IMPROVEMENTS:
1. Ehlers Fisher Transform (period=9) - catches reversals in bear/range markets
2. 1d HMA for macro bias (only long when price>1d HMA, only short when price<1d HMA)
3. 1w HMA for ultra-long-term trend (avoid counter-trend trades)
4. Volume confirmation (lenient: >0.5x avg) to avoid illiquid entries
5. ATR trailing stop at 2.5x for risk management
6. Looser Fisher thresholds: cross above -1.5 (long), cross below +1.5 (short)
7. Discrete position sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

TARGET: 35-55 trades/year, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_hma_volume_1d1w_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Converts price into a Gaussian normal distribution.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher_prev = np.zeros(n)
    
    # Typical price
    typical = (high + low) / 2.0
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(typical[i-period+1:i+1])
        lowest = np.min(typical[i-period+1:i+1])
        
        # Normalize price to range -1 to +1
        range_val = highest - lowest
        if range_val < 1e-10:
            range_val = 1e-10
        
        normalized = 0.999 * (2.0 * (typical[i] - lowest) / range_val - 1.0)
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Previous Fisher (for crossover detection)
        if i > period:
            fisher_prev[i] = fisher[i-1]
        else:
            fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

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
    
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Directional Movement
    plus_dm = np.maximum(high_s - high_s.shift(1), 0).values
    minus_dm = np.maximum(low_s.shift(1) - low, 0).values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Smooth with Wilder's method (EMA with span=period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    # DX and ADX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx = np.nan_to_num(adx, nan=0.0)
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # Volume average (20-bar)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1w HMA for ultra-long-term trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(fisher[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        # === HTF MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === FISHER TRANSFORM CROSSOVERS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_cross_long = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_cross_short = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # === VOLUME FILTER (lenient) ===
        volume_ok = volume[i] > 0.5 * vol_avg[i]
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 20.0
        
        # === ENTRY LOGIC (simple OR logic to ensure trades fire) ===
        new_signal = 0.0
        
        # LONG entries
        # Trigger 1: Fisher cross + price above 1d HMA (bullish bias)
        long_trigger_1 = fisher_cross_long and price_above_hma_1d and volume_ok
        # Trigger 2: Fisher cross + price above 1w HMA (strong bullish bias)
        long_trigger_2 = fisher_cross_long and price_above_hma_1w and volume_ok
        # Trigger 3: RSI oversold + price above 1d HMA (mean reversion in uptrend)
        long_trigger_3 = (rsi_14[i] < 35.0) and price_above_hma_1d and volume_ok
        
        if long_trigger_1 or long_trigger_2 or long_trigger_3:
            if price_above_hma_1w:
                new_signal = POSITION_SIZE_FULL  # Strong bullish bias
            else:
                new_signal = POSITION_SIZE_HALF  # Weaker bias
        
        # SHORT entries
        # Trigger 1: Fisher cross + price below 1d HMA (bearish bias)
        short_trigger_1 = fisher_cross_short and price_below_hma_1d and volume_ok
        # Trigger 2: Fisher cross + price below 1w HMA (strong bearish bias)
        short_trigger_2 = fisher_cross_short and price_below_hma_1w and volume_ok
        # Trigger 3: RSI overbought + price below 1d HMA (mean reversion in downtrend)
        short_trigger_3 = (rsi_14[i] > 65.0) and price_below_hma_1d and volume_ok
        
        if short_trigger_1 or short_trigger_2 or short_trigger_3:
            if price_below_hma_1w:
                new_signal = -POSITION_SIZE_FULL  # Strong bearish bias
            else:
                new_signal = -POSITION_SIZE_HALF  # Weaker bias
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and trend still valid (don't exit on every bar)
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price still above 1d HMA
                if price_above_hma_1d:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price still below 1d HMA
                if price_below_hma_1d:
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
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price crosses below 1d HMA (trend changed)
        if in_position and position_side > 0 and price_below_hma_1d:
            new_signal = 0.0
        
        # Exit short if price crosses above 1d HMA (trend changed)
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
                # Position flip
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