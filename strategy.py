#!/usr/bin/env python3
"""
Experiment #026: 6h Supertrend + 1d SMA200 + Volume + Choppiness Regime

HYPOTHESIS: Supertrend (ATR-based, multiplier 3) provides cleaner trend signals
than EMA crossovers. Combined with 1d SMA200 for trend alignment, volume 
confirmation for institutional conviction, and Choppiness regime filter, this
captures trend starts without overtrading.

WHY 6h: Between 4h (too many trades) and 12h (too few). Supertrend on 6h 
captures multi-day trends with ~30-60 trades/year.

KEY DIFFERENCE FROM FAILURES:
- NOT Elder Ray (failed #012, #020)
- NOT pure Donchian (overtrades)
- Supertrend is continuous ATR-based, not oscillating

TARGET: 75-150 total trades over 4 years = 19-37/year. HARD MAX: 300.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_supertrend_1d_sma200_vol_chop_v1"
timeframe = "6h"
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

def calculate_supertrend(high, low, close, atr_period=10, multiplier=3.0):
    """
    Supertrend indicator
    Returns: supertrend values (positive = bull, negative = bear), upper_band, lower_band
    """
    atr = calculate_atr(high, low, close, period=atr_period)
    n = len(close)
    
    # hl2 = (high + low) / 2
    hl2 = (high + low) / 2.0
    
    # Upper and lower bands
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Supertrend
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = bull, -1 = bear
    
    for i in range(n):
        if i == 0:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            # Previous values
            prev_st = supertrend[i-1]
            prev_dir = direction[i-1]
            prev_close = close[i-1]
            
            if pd.isna(atr[i]) or atr[i] <= 0:
                supertrend[i] = prev_st
                direction[i] = prev_dir
                continue
            
            # Current bands
            curr_upper = upper_band[i]
            curr_lower = lower_band[i]
            
            if prev_dir == 1:  # Was bullish
                # Check if should flip to bearish
                if close[i] < prev_st:
                    supertrend[i] = curr_upper
                    direction[i] = -1
                else:
                    # Stay bullish, lower band can't go below previous
                    supertrend[i] = max(prev_st, curr_lower)
                    direction[i] = 1
            else:  # Was bearish
                # Check if should flip to bullish
                if close[i] > prev_st:
                    supertrend[i] = curr_lower
                    direction[i] = 1
                else:
                    # Stay bearish, upper band can't go above previous
                    supertrend[i] = min(prev_st, curr_upper)
                    direction[i] = -1
    
    return supertrend, direction, upper_band, lower_band

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range (avoid trend following)
    CHOP < 38.2 = trending (trend following works)
    """
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_rsi(close, period=14):
    """RSI indicator"""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction
    sma_200_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # Local 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    supertrend, direction, upper_band, lower_band = calculate_supertrend(
        high, low, close, atr_period=10, multiplier=3.0
    )
    chop = calculate_choppiness_index(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Track Supertrend flips for entry signals
    supertrend_dir_change = np.zeros(n)
    for i in range(1, n):
        if direction[i] != direction[i-1]:
            supertrend_dir_change[i] = direction[i]  # +1 = bull flip, -1 = bear flip
    
    signals = np.zeros(n)
    SIZE = 0.30  # Standard sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 250  # Need 200 for SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(sma_200_aligned[i]):
            continue
        if np.isnan(chop[i]):
            continue
        
        # === MARKET REGIME ===
        # Only trend follow when CHOP < 61.8 (trending)
        # In choppy markets (CHOP > 61.8), we stay flat
        is_trending = chop[i] < 61.8
        is_choppy = chop[i] > 61.8
        
        # === TREND DIRECTION (1d SMA200) ===
        price_above_1d_sma = close[i] > sma_200_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === ENTRY: Supertrend flip + trend alignment + volume ===
            
            # LONG: Supertrend flipped to bullish (+1), price above 1d SMA, trending regime, volume
            if supertrend_dir_change[i] == 1:
                if price_above_1d_sma and is_trending and vol_spike:
                    desired_signal = SIZE
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    entry_atr = atr_14[i]
                    entry_bar = i
            
            # SHORT: Supertrend flipped to bearish (-1), price below 1d SMA, trending regime, volume
            elif supertrend_dir_change[i] == -1:
                if not price_above_1d_sma and is_trending and vol_spike:
                    desired_signal = -SIZE
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    entry_atr = atr_14[i]
                    entry_bar = i
        
        # === STOPLOSS CHECK (2.0 ATR trailing) ===
        if in_position:
            if position_side > 0:
                # Long stop: price drops below Supertrend line
                if close[i] < supertrend[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                # RSI overbought exit
                elif rsi_14[i] > 78:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
            
            elif position_side < 0:
                # Short stop: price rises above Supertrend line
                if close[i] > supertrend[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                # RSI oversold exit
                elif rsi_14[i] < 22:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        signals[i] = desired_signal
    
    return signals