#!/usr/bin/env python3
"""
Experiment #028: 12h Bollinger Band Mean Reversion + Williams %R + Choppiness

HYPOTHESIS: When price reaches Bollinger Band extremes on the 12h chart,
institutional moves often reverse. Williams %R confirms momentum exhaustion,
and Choppiness Index keeps us out of trending markets where BB mean reversion fails.

WHY 12h: Slow enough to reduce fee drag (target 30-60 trades/year), fast enough
to capture mean reversion cycles. BB squeeze expansions on 12h represent multi-day
institutional positioning.

WHY IT WORKS IN BULL AND BEAR: 
- In BULL: Corrections to lower BB often reverse (buy dips)
- In BEAR: Rallies to upper BB reverse (fade rallies)
- Uses 1d SMA200 to determine bias: above = primarily long reversion, below = short reversion

TARGET: 50-150 total trades over 4 years = 12-37/year. HARD MAX: 200.
Signal size: 0.25-0.30 (discrete levels).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_bb_wr_chop_1d_v1"
timeframe = "12h"
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

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum indicator"""
    n = len(close)
    wr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high != lowest_low:
            wr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return wr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend bias
    sma_200 = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200)
    
    # Local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    wr = calculate_williams_r(high, low, close, period=14)
    
    # Bollinger Bands (20 period = 10 days on 12h)
    bb_period = 20
    bb_std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std(ddof=0).values
    bb_mid = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / np.where(bb_mid > 0, bb_mid, 1)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = -0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 300  # Need 200 for SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_200_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(wr[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA200) ===
        # Strict trend filter - only trade with trend
        price_above_200_sma = close[i] > sma_200_aligned[i]
        price_below_200_sma = close[i] < sma_200_aligned[i]
        
        # === REGIME (Choppiness Index) ===
        # Only trade in choppy markets (CHOP > 50) for mean reversion
        # Skip trending markets (CHOP < 40) - trend following works better there
        is_choppy_enough = chop[i] > 50.0
        is_too_choppy = chop[i] > 65.0  # Skip extreme chop
        
        # Skip if too trending or too choppy
        if (is_too_choppy or chop[i] < 40.0) and not in_position:
            signals[i] = 0.0
            continue
        
        # === WILLIAMS % R SIGNALS ===
        # Oversold: WR < -80, Overbought: WR > -20
        is_oversold = wr[i] < -80
        is_overbought = wr[i] > -20
        
        # === BOLLINGER BAND SIGNALS ===
        # Price at lower band = potential support
        # Price at upper band = potential resistance
        price_at_lower_bb = close[i] <= bb_lower[i] * 1.02  # Within 2% of lower band
        price_at_upper_bb = close[i] >= bb_upper[i] * 0.98  # Within 2% of upper band
        
        # BB squeeze detection - low volatility often precedes moves
        bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
        is_squeeze = bb_width[i] < bb_width_ma[i] * 0.7 if not np.isnan(bb_width_ma[i]) else False
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Price at lower BB + oversold Williams %R ===
            # Only in choppy markets, with trend alignment
            if price_at_lower_bb and is_oversold and is_choppy_enough:
                if price_above_200_sma:  # With trend
                    desired_signal = SIZE_LONG
                elif not is_too_choppy and price_below_200_sma:  # Counter-trend only in moderate chop
                    desired_signal = SIZE_LONG * 0.5  # Half size for counter-trend
            
            # === SHORT: Price at upper BB + overbought Williams %R ===
            if price_at_upper_bb and is_overbought and is_choppy_enough:
                if price_below_200_sma:  # With trend
                    desired_signal = SIZE_SHORT
                elif not is_too_choppy and price_above_200_sma:  # Counter-trend only in moderate chop
                    desired_signal = SIZE_SHORT * 0.5  # Half size for counter-trend
        
        # === STOPLOSS CHECK (2.5 ATR stop) ===
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
        
        # === MEAN REVERSION TARGET (take profit at mid BB) ===
        if in_position:
            bars_held = i - entry_bar
            mid_reached_long = position_side > 0 and close[i] >= bb_mid[i]
            mid_reached_short = position_side < 0 and close[i] <= bb_mid[i]
            
            # Take profit when price returns to mid-BB (after at least 2 bars)
            if bars_held >= 2 and (mid_reached_long or mid_reached_short):
                desired_signal = 0.0
            
            # Stop if we go deeper into the band (mean reversion failed)
            if bars_held >= 4:
                if position_side > 0 and close[i] < bb_lower[i] * 0.98:
                    desired_signal = 0.0
                if position_side < 0 and close[i] > bb_upper[i] * 1.02:
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