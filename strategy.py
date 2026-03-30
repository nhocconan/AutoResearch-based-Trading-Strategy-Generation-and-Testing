#!/usr/bin/env python3
"""
Experiment #028: 12h Donchian Breakout + 1w HMA Trend + Volume Spike

HYPOTHESIS: This combines proven elements from DB winners:
- 12h timeframe (lower trade frequency, ~25-50/year vs 50-100 on 4h)
- 1w HMA(21) for primary trend (proven in mtf_1d_kama_rsi_chop_regime_1w_v1: Sharpe 1.31)
- Donchian(20) breakout for price structure (proven in multiple SOL winners)
- Volume spike confirmation (2.0x threshold - strict)
- Choppiness filter to avoid range-bound whipsaws

KEY IMPROVEMENTS over failed attempts:
- Use CLOSE > previous Donchian high (not intrabar high) = cleaner signal
- 1w HMA instead of 1d SMA = longer trend filter, fewer trades
- STRICTER volume threshold (2.0x) to reduce false breakouts
- Mandatory 4-bar hold minimum to avoid chop
- Target: 50-100 total trades over 4 years

WHY IT WORKS IN BULL AND BEAR:
- Long entries only when 1w trend is UP (bull) + breakout (momentum)
- Short entries only when 1w trend is DOWN (bear) + breakdown (momentum)
- Symmetrical Donchian channels work in both directions
- Choppiness keeps us out of low-volatility environments
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_1w_hma_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(data, period):
    """Hull Moving Average"""
    half_length = period // 2
    sqrt_length = int(np.sqrt(period))
    
    # Calculate WMA parts
    wma_half = pd.Series(data).rolling(window=half_length, min_periods=half_length).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    )
    wma_full = pd.Series(data).rolling(window=period, min_periods=period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    )
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    hma_raw = 2 * wma_half - wma_full
    hma = hma_raw.rolling(window=sqrt_length, min_periods=sqrt_length).mean()
    
    return hma.values

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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - lower = trending, higher = choppy"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w HMA(21) for primary trend direction
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Local indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Donchian channels (24 periods = 12 days on 12h - slightly longer for fewer trades)
    donchian_period = 24
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume average (30 bars for stability on 12h)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = max(100, donchian_period + 10)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1w HMA21) ===
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        
        # === REGIME (Choppiness Index) ===
        # Only trade when trending (CHOP < 50)
        # In choppy markets (CHOP > 61.8), stay flat
        is_trending = chop[i] < 50.0
        is_choppy = chop[i] > 61.8
        
        # Skip if too choppy
        if is_choppy and not in_position:
            signals[i] = 0.0
            continue
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Use CLOSE > previous Donchian high/low (cleaner than intrabar high/low)
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        
        # Volume confirmation (strict: 2.0x average)
        vol_spike = vol_ratio[i] > 2.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above previous Donchian high ===
            # Price CLOSES above previous 24-bar high with volume + trend
            if close[i] > prev_donchian_high and price_above_1w_hma and is_trending:
                if vol_spike:  # Strong volume confirmation required
                    desired_signal = SIZE
            
            # === SHORT: Breakdown below previous Donchian low ===
            # Price CLOSES below previous 24-bar low with volume + trend
            if close[i] < prev_donchian_low and not price_above_1w_hma and is_trending:
                if vol_spike:  # Strong volume confirmation required
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing - slightly wider) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TIME-BASED EXIT (hold at least 4 bars = 2 days) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 4:
            # Exit if price reverts to middle of channel
            channel_mid = (donchian_high[i] + donchian_low[i]) / 2
            if position_side > 0 and close[i] < channel_mid:
                desired_signal = 0.0
            if position_side < 0 and close[i] > channel_mid:
                desired_signal = 0.0
        
        # === ANTI-CHOP EXIT ===
        # Exit if choppiness rises sharply while in position
        if in_position and not is_trending and bars_held >= 2:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
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
        
        signals[i] = desired_signal
    
    return signals