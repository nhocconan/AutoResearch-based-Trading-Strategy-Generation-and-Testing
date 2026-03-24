#!/usr/bin/env python3
"""
Experiment #240: 6h Primary + 1d/1w HTF — Fisher Transform Reversal + Multi-TF Trend

Hypothesis: 6h timeframe with Ehlers Fisher Transform can catch reversals in bear/range
markets (2022 crash, 2025 bear) while using 1d/1w HMA for trend bias. Previous 6h Fisher
attempt (#231) failed with 0 trades - likely entry conditions too strict.

Key improvements over #231:
- Looser Fisher thresholds (-1.5/+1.5 instead of -2.0/+2.0)
- Use 1d HMA(50) for intermediate trend, 1w HMA(21) for major bias
- Only require ONE HTF alignment (not both) to generate trades
- Add volume confirmation to filter false breakouts

Fisher Transform:
- Period=9, converts price to near-Gaussian distribution
- Long: Fisher crosses above -1.5 (oversold reversal)
- Short: Fisher crosses below +1.5 (overbought reversal)
- Proven to work in bear/range markets (2022, 2025)

HTF Filters:
- 1d HMA(50): price above = bullish bias, below = bearish bias
- 1w HMA(21): major trend direction (only require alignment on one TF)

Position sizing: 0.25 base, 0.30 strong signals
Stoploss: 2.5x ATR trailing

Target: Sharpe>0.399 (beat current 6h best), DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_reversal_hma_1d1w_v2"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - converts price to Gaussian distribution
    Highlights turning points with sharp peaks
    
    Formula:
    1. Calculate typical price: (High + Low) / 2
    2. Normalize: (price - lowest) / (highest - lowest) over period
    3. Transform: 0.5 * ln((1 + x) / (1 - x)) where x = 2*normalized - 1
    4. Smooth with EMA
    """
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        # Normalize price over lookback period
        highest = np.max(typical[i-period+1:i+1])
        lowest = np.min(typical[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range < 1e-10:
            continue
        
        # Normalize to 0-1 range
        normalized = (typical[i] - lowest) / price_range
        
        # Convert to -1 to +1 range
        x = 2.0 * normalized - 1.0
        
        # Clamp to avoid division by zero
        x = np.clip(x, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        # Trigger line (1-period EMA of fisher)
        if i > period:
            trigger[i] = 0.5 * fisher[i] + 0.5 * fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs moving average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_ma
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for intermediate trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    atr = calculate_atr(high, low, close, period=14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, period=9)
    rsi = calculate_rsi(close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # 6h HMA for local trend confirmation
    hma_6h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25  # 25% base position size
    SIZE_STRONG = 0.30  # 30% for strong signals
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d and 1w HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Require at least ONE HTF alignment (not both - too restrictive)
        htf_bullish = htf_1d_bull or htf_1w_bull
        htf_bearish = htf_1d_bear or htf_1w_bear
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_cross_up = False
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(fisher_trigger[i-1]):
            fisher_cross_up = (fisher_trigger[i-1] <= -1.5 and fisher[i] > -1.5)
        
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_cross_down = False
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(fisher_trigger[i-1]):
            fisher_cross_down = (fisher_trigger[i-1] >= 1.5 and fisher[i] < 1.5)
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = False
        if not np.isnan(vol_ratio[i]):
            vol_confirmed = vol_ratio[i] > 0.8  # At least 80% of avg volume
        
        # === 6h HMA LOCAL TREND ===
        hma_6h_bull = close[i] > hma_6h[i]
        hma_6h_bear = close[i] < hma_6h[i]
        
        # === RSI FILTER (avoid extreme overbought/oversold against trade) ===
        rsi_ok_long = True
        rsi_ok_short = True
        if not np.isnan(rsi[i]):
            rsi_ok_long = rsi[i] < 70  # Not extremely overbought for long
            rsi_ok_short = rsi[i] > 30  # Not extremely oversold for short
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: Fisher reversal + HTF bias + volume
        if fisher_cross_up and htf_bullish and vol_confirmed and rsi_ok_long:
            # Strong signal if 6h HMA also bullish
            if hma_6h_bull:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT ENTRY: Fisher reversal + HTF bias + volume
        elif fisher_cross_down and htf_bearish and vol_confirmed and rsi_ok_short:
            # Strong signal if 6h HMA also bearish
            if hma_6h_bear:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals