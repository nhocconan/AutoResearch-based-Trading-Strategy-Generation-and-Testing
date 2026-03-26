#!/usr/bin/env python3
"""
Experiment #021: 4h Williams %R + Donchian + Choppiness Regime

HYPOTHESIS: Williams %R captures oversold/overbought extremes with built-in lookback,
making it ideal for mean-reversion at key turning points. Combined with 20-period
Donchian channel (structure), Choppiness Index (regime filter), and volume confirmation,
this targets 75-200 total trades over 4 years.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR MARKETS:
- BULL: Long when %R <-80 + price > SMA200 + break upper Donchian + volume spike
       Captures dips during uptrend.
- BEAR: Short when %R >-20 + price < SMA200 + break lower Donchian + volume spike
       Captures rallies during downtrend.
- CHOP FILTER: Skip entries when CHOP > 61.8 (range-bound = don't chase)
- VOLUME: Confirms institutional interest at extremes

This is a REFINEMENT of proven CRSI+Donchian pattern, simpler and more robust.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_williams_r_donchian_chop_v1"
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

def calculate_williams_r(high, low, close, period=14):
    """
    Williams %R
    Values: 0 to -100
    > -20 = overbought
    < -80 = oversold
    """
    n = len(close)
    willr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high != lowest_low:
            willr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return willr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - 20 period as default"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        upper[i] = hh
        lower[i] = ll
        middle[i] = (hh + ll) / 2.0
    
    return upper, middle, lower

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range market (mean reversion works)
    CHOP < 38.2 = trending market (momentum works)
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # 1d EMA48 for medium trend
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=48, min_periods=48, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    willr = calculate_williams_r(high, low, close, period=14)
    upper_dc, middle_dc, lower_dc = calculate_donchian(high, low, period=20)
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # Volume for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 250  # Need 200 for SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(willr[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(upper_dc[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA200) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # === REGIME (Choppiness Index) ===
        # In trending market (CHOP < 38.2), use momentum
        # In range market (CHOP > 61.8), use mean reversion
        is_trending = chop[i] < 38.2
        is_choppy = chop[i] > 61.8
        
        # === DONCHIAN CHANNEL LEVELS ===
        dc_upper = upper_dc[i]
        dc_lower = lower_dc[i]
        dc_middle = middle_dc[i]
        
        # === WILLIAMS %R ===
        willr_val = willr[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === TRENDING MARKET: Follow Donchian breakouts ===
            if is_trending:
                # Long: Break above upper Donchian + oversold recovery + bullish trend
                if price_above_1d_sma and price_above_1d_ema:
                    # Price breaking out of channel
                    if close[i] > dc_upper and vol_spike:
                        desired_signal = SIZE
                    # OR: Price at lower channel + Williams %R oversold
                    elif close[i] < dc_lower * 1.02 and willr_val < -80:
                        if vol_spike:
                            desired_signal = SIZE
                    # OR: Williams %R deeply oversold + recovering + above middle
                    elif willr_val > -50 and willr_val < -20 and close[i] > dc_middle:
                        if vol_spike:
                            desired_signal = SIZE
                
                # Short: Break below lower Donchian + overbought + bearish trend
                if not price_above_1d_sma and not price_above_1d_ema:
                    # Price breaking down from channel
                    if close[i] < dc_lower and vol_spike:
                        desired_signal = -SIZE
                    # OR: Price at upper channel + Williams %R overbought
                    elif close[i] > dc_upper * 0.98 and willr_val > -20:
                        if vol_spike:
                            desired_signal = -SIZE
                    # OR: Williams %R deeply overbought + collapsing + below middle
                    elif willr_val < -50 and willr_val > -80 and close[i] < dc_middle:
                        if vol_spike:
                            desired_signal = -SIZE
            
            # === CHOPPY MARKET: Fade Donchian extremes ===
            if is_choppy:
                # Long: Price at lower Donchian + oversold + volume
                if close[i] <= dc_lower * 1.03 and willr_val < -85 and price_above_1d_sma:
                    desired_signal = SIZE
                
                # Short: Price at upper Donchian + overbought + volume
                if close[i] >= dc_upper * 0.97 and willr_val > -15 and not price_above_1d_sma:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * atr_14[i]
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * atr_14[i]
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === MINIMUM HOLD (6 bars = 1 day) ===
        bars_held = i - entry_bar if in_position else 0
        min_hold_bars = 6
        
        if in_position and bars_held >= min_hold_bars:
            # Exit on Williams %R reversal
            if position_side > 0 and willr_val > -10:
                desired_signal = 0.0
            if position_side < 0 and willr_val < -90:
                desired_signal = 0.0
        
        # === TIME-BASED EXIT (max hold = 20 bars = 5 days) ===
        max_hold_bars = 20
        if in_position and bars_held >= max_hold_bars:
            desired_signal = 0.0
        
        # === ATR-BASED PROFIT TARGET ===
        if in_position:
            profit_target_mult = 3.0
            if position_side > 0:
                profit_target = entry_price + profit_target_mult * atr_14[i]
                if high[i] >= profit_target:
                    desired_signal = SIZE / 2  # Take partial profit
            if position_side < 0:
                profit_target = entry_price - profit_target_mult * atr_14[i]
                if low[i] <= profit_target:
                    desired_signal = -SIZE / 2  # Take partial profit
        
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
                # Same direction - maintain/reduce position
                pass
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