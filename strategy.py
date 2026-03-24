#!/usr/bin/env python3
"""
Experiment #603: 6h Primary + 1d/1w HTF — Donchian Breakout + BB Squeeze + Volume Confirm

Hypothesis: 6h timeframe is ideal for swing trades (2-5 day holds). Donchian Channel
breakouts capture multi-day momentum moves, but need filters to avoid false breakouts.
Key innovations vs failed 6h strategies:
1. Donchian(20) breakout with volume confirmation (volume > 1.5x avg)
2. Bollinger Band squeeze filter (BW percentile < 20% = coiling before breakout)
3. 1d HMA(21) for medium trend bias + 1w HMA(21) for macro bias
4. Asymmetric sizing: full size with HTF trend, half size against
5. ATR(14)*2.5 stoploss with trailing on profits
6. Exit on opposite Donchian break or RSI extreme

Why this should work on 6h:
- 6h captures multi-day swings better than 4h (less noise) and 12h (more signals)
- Donchian breakouts work well for swing trading (2-5 day holds)
- BB squeeze identifies low-vol coiling before explosive moves
- Volume filter reduces false breakouts (common failure mode)
- HTF bias prevents trading against major trend

Target: 30-50 trades/year, Sharpe>0.40, DD<-30%
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_bb_squeeze_vol_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - highest high and lowest low over period
    Returns: upper_band, lower_band, middle_band
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i - period + 1:i + 1])
        lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
    middle = (upper + lower) / 2.0
    return upper, lower, middle

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """
    Bollinger Bands with bandwidth calculation
    Returns: upper, lower, middle, bandwidth
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    bandwidth = (upper - lower) / middle * 100.0
    
    return upper, lower, middle, bandwidth

def calculate_bw_percentile(bandwidth, lookback=60):
    """
    Bollinger Band Width percentile over lookback period
    Low percentile (<20%) = squeeze (coiling before breakout)
    """
    n = len(bandwidth)
    percentile = np.zeros(n)
    percentile[:] = np.nan
    
    for i in range(lookback - 1, n):
        if not np.isnan(bandwidth[i]):
            window = bandwidth[i - lookback + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                percentile[i] = np.sum(valid < bandwidth[i]) / len(valid) * 100.0
            else:
                percentile[i] = 50.0
    
    return percentile

def calculate_volume_ratio(volume, period=20):
    """
    Volume ratio vs moving average
    Ratio > 1.5 = above average volume (confirmation)
    """
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_ma
    
    vol_ratio[:period] = np.nan
    return vol_ratio

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for medium trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    bb_upper, bb_lower, bb_mid, bb_bandwidth = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bb_percentile = calculate_bw_percentile(bb_bandwidth, lookback=60)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    SIZE_WEAK = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    prev_donchian_break = 0  # Track last breakout direction
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(bb_percentile[i]) or np.isnan(vol_ratio[i]):
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
        
        # === HTF BIAS (1w macro + 1d medium) ===
        htf_bull = close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # === DONCHIAN BREAKOUT DETECTION ===
        breakout_long = close[i] > donchian_upper[i-1] and close[i-1] <= donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1] and close[i-1] >= donchian_lower[i-1]
        
        # === BB SQUEEZE FILTER ===
        bb_squeeze = bb_percentile[i] < 25.0  # Bottom 25% = coiling
        bb_expanding = bb_percentile[i] > bb_percentile[i-5] if i >= 5 and not np.isnan(bb_percentile[i-5]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.3  # 30% above average
        
        # === RSI FILTER (avoid extreme overbought/oversold breakouts) ===
        rsi_valid_long = rsi[i] < 70.0  # Not extremely overbought
        rsi_valid_short = rsi[i] > 30.0  # Not extremely oversold
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: Donchian breakout + squeeze + volume + HTF bias
        if breakout_long and rsi_valid_long:
            if htf_bull and bb_squeeze and vol_confirm:
                desired_signal = SIZE_STRONG  # All conditions aligned
            elif htf_bull and (bb_squeeze or vol_confirm):
                desired_signal = SIZE_BASE  # Partial confirmation
            elif htf_neutral and bb_squeeze and vol_confirm:
                desired_signal = SIZE_BASE  # Neutral HTF but strong setup
            elif bb_squeeze and vol_confirm:
                desired_signal = SIZE_WEAK  # Weak signal without HTF
        
        # SHORT ENTRY: Donchian breakdown + squeeze + volume + HTF bias
        elif breakout_short and rsi_valid_short:
            if htf_bear and bb_squeeze and vol_confirm:
                desired_signal = -SIZE_STRONG  # All conditions aligned
            elif htf_bear and (bb_squeeze or vol_confirm):
                desired_signal = -SIZE_BASE  # Partial confirmation
            elif htf_neutral and bb_squeeze and vol_confirm:
                desired_signal = -SIZE_BASE  # Neutral HTF but strong setup
            elif bb_squeeze and vol_confirm:
                desired_signal = -SIZE_WEAK  # Weak signal without HTF
        
        # === EXIT CONDITIONS (opposite breakout or RSI extreme) ===
        if in_position and position_side > 0:
            # Exit long on opposite breakout or RSI overbought
            if breakout_short or rsi[i] > 75.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short on opposite breakout or RSI oversold
            if breakout_long or rsi[i] < 25.0:
                desired_signal = 0.0
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            # Trail stop on profits
            if highest_since_entry > entry_price + 1.5 * entry_atr:
                trailing_stop = highest_since_entry - 2.5 * entry_atr
                stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            # Trail stop on profits
            if lowest_since_entry < entry_price - 1.5 * entry_atr:
                trailing_stop = lowest_since_entry + 2.5 * entry_atr
                stop_price = min(stop_price, trailing_stop)
        
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
        elif desired_signal >= SIZE_WEAK * 0.9:
            final_signal = SIZE_WEAK
        elif desired_signal <= -SIZE_WEAK * 0.9:
            final_signal = -SIZE_WEAK
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals