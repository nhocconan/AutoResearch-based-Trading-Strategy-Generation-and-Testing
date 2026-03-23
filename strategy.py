#!/usr/bin/env python3
"""
Experiment #430: 1h Primary + 4h/12h HTF — RSI Pullback + HMA Trend + Volume

Hypothesis: 1h timeframe with 4h/12h trend bias should generate 30-60 trades/year
with better entry timing than 4h strategies. Learning from 11 straight 0-trade failures:
1. LOOSEN entry conditions - use OR logic, not strict AND
2. Remove session filter (was blocking trades)
3. Wider RSI ranges (30-55 for long, 45-70 for short)
4. Volume filter at 0.7x (not 0.8x)
5. Multiple entry paths - any one can trigger

Key components:
1. 4h HMA(21) for primary trend direction
2. 12h HMA(21) for overall market bias (bull/bear filter)
3. 1h RSI(14) pullback entries within HTF trend
4. Volume filter (volume > 0.7x 20-period avg)
5. ATR(14) trailing stoploss at 2.5x for risk management

Target: Sharpe > 0.612, 120-240 trades over 4-year train, DD < -40%
CRITICAL: Must generate trades! Looser conditions than recent failures.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_hma_volume_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
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

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1h indicators (primary timeframe)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align HTF HMA for bias (4h and 12h)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[100:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% position size for 1h
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(rsi_14[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(vol_sma_20[i]):
            continue
        
        # === TREND BIAS (12h HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === PRIMARY TREND (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === RSI PULLBACK SIGNALS (LOOSENED for trade frequency) ===
        # Long: RSI pulled back to 30-55 in uptrend
        rsi_long_pullback = 30.0 <= rsi_14[i] <= 55.0
        # Short: RSI rallied to 45-70 in downtrend
        rsi_short_pullback = 45.0 <= rsi_14[i] <= 70.0
        
        # RSI extreme for mean reversion
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === VOLUME FILTER (LOOSENED) ===
        vol_ratio = volume[i] / (vol_sma_20[i] + 1e-10)
        volume_ok = vol_ratio > 0.7  # At least 70% of average volume
        
        # === VOL FILTER (ATR based) ===
        vol_ratio_atr = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio_atr > 2.5:
            position_size = BASE_SIZE * 0.5
        elif vol_ratio_atr > 1.8:
            position_size = BASE_SIZE * 0.7
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP — Multiple confluence conditions (ANY ONE can trigger)
        long_bias = price_above_hma_12h or price_above_hma_4h
        
        if long_bias:
            # Condition 1: RSI pullback in uptrend + volume
            if rsi_long_pullback and volume_ok:
                desired_signal = position_size
            # Condition 2: RSI oversold (strong mean reversion) + volume
            elif rsi_oversold and volume_ok:
                desired_signal = position_size
            # Condition 3: Price above SMA200 + RSI < 50
            elif close[i] > sma_200[i] and rsi_14[i] < 50.0:
                desired_signal = position_size * 0.7
        
        # SHORT SETUP — Multiple confluence conditions (ANY ONE can trigger)
        short_bias = price_below_hma_12h or price_below_hma_4h
        
        if short_bias:
            # Condition 1: RSI pullback in downtrend + volume
            if rsi_short_pullback and volume_ok:
                desired_signal = -position_size
            # Condition 2: RSI overbought (strong mean reversion) + volume
            elif rsi_overbought and volume_ok:
                desired_signal = -position_size
            # Condition 3: Price below SMA200 + RSI > 50
            elif close[i] < sma_200[i] and rsi_14[i] > 50.0:
                desired_signal = -position_size * 0.7
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === RSI EXTREME EXIT ===
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
            desired_signal = 0.0
        
        # === HTF BIAS REVERSAL EXIT ===
        if in_position and position_side > 0 and price_below_hma_12h and price_below_hma_4h:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_12h and price_above_hma_4h:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (price_above_hma_12h or price_above_hma_4h):
                desired_signal = position_size
            elif position_side < 0 and (price_below_hma_12h or price_below_hma_4h):
                desired_signal = -position_size
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals