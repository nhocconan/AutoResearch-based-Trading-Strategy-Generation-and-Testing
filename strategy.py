#!/usr/bin/env python3
"""
Experiment #024: 6h Supertrend + Volume Spike + Williams %R Regime

HYPOTHESIS: Supertrend provides adaptive trailing stops that work in both
bull and bear markets. The Williams %R (not RSI or ADX) is used as a
regime filter - when %R < -80 (deep oversold), mean reversion is likely.
When %R > -20 (overbought), reversions are also likely. This captures
"exhaustion" points rather than trend continuation.

DIFFERENT FROM PREVIOUS:
- Not Donchian breakout (chases price)
- Not Camarilla fade (too few trades historically)
- Supertrend trails with ATR bands (adaptive)
- Williams %R for regime (not RSI/ADX/chop)

TIMEFRAME: 6h primary, 1d for pivot confirmation
TRADE COUNT: Target 100-200 total over 4 years (25-50/year)
SIZE: 0.30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_supertrend_vol_williams_v1"
timeframe = "6h"
leverage = 1.0

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Supertrend indicator using ATR bands.
    Returns: supertrend (1=bullish, -1=bearish), upper_band, lower_band
    """
    n = len(close)
    
    # Calculate ATR
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate bands
    hl_avg = (high + low) / 2.0
    upper_band = hl_avg + multiplier * atr
    lower_band = hl_avg - multiplier * atr
    
    # Supertrend calculation
    supertrend = np.zeros(n)
    supertrend[0] = 1  # Start bullish
    
    for i in range(1, n):
        # Previous values
        prev_close = close[i-1]
        prev_st = supertrend[i-1]
        prev_upper = upper_band[i-1]
        prev_lower = lower_band[i-1]
        
        if prev_st == 1:  # Was bullish
            if close[i] < prev_upper:
                supertrend[i] = -1
            else:
                supertrend[i] = 1
                lower_band[i] = max(lower_band[i], prev_lower)
        else:  # Was bearish
            if close[i] > prev_lower:
                supertrend[i] = 1
            else:
                supertrend[i] = -1
                upper_band[i] = min(upper_band[i], prev_upper)
    
    return supertrend, upper_band, lower_band, atr

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum indicator for oversold/overbought"""
    n = len(close)
    result = np.full(n, np.nan)
    
    for i in range(period-1, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest != lowest:
            result[i] = -100 * (highest - close[i]) / (highest - lowest)
    
    return result

def calculate_camarilla_pivots(high, low, close):
    """Camarilla pivot levels for structure confirmation"""
    rng = high - low
    r3 = close + rng * 1.1
    s3 = close - rng * 1.1
    return r3, s3

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Volume spike detection"""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    return ratio > threshold

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d for pivot confirmation (call ONCE) ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Camarilla pivots for structure
    r3_1d, s3_1d = calculate_camarilla_pivots(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # 1d EMA 50 for trend
    ema_1d_50 = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # === 6h indicators ===
    supertrend, upper_band, lower_band, atr = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    williams_r = calculate_williams_r(high, low, close, period=14)
    vol_spike = calculate_volume_spike(volume, period=20, threshold=1.5)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    supertrend_at_entry = 0
    
    warmup = 50  # Need enough for indicators
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION via Williams %R ===
        wr = williams_r[i]
        is_oversold = wr < -80  # Deep oversold - potential bounce
        is_overbought = wr > -20  # Overbought - potential drop
        is_neutral = not is_oversold and not is_overbought
        
        # === TREND via 1d EMA ===
        uptrend = close[i] > ema_aligned[i]
        downtrend = close[i] < ema_aligned[i]
        
        # === PIVOT PROXIMITY ===
        # Price near 1d S3 = potential long
        near_1d_s3 = close[i] < s3_aligned[i] * 1.02 if not np.isnan(s3_aligned[i]) else False
        # Price near 1d R3 = potential short
        near_1d_r3 = close[i] > r3_aligned[i] * 0.98 if not np.isnan(r3_aligned[i]) else False
        
        # === SUPERTREND SIGNALS ===
        st_bullish = supertrend[i] == 1
        st_bearish = supertrend[i] == -1
        
        # === MINIMUM HOLD: 2 bars (12h) to avoid fee churn ===
        min_hold = (i - entry_bar) >= 2
        
        # === ATR STOP LOSS ===
        def check_stop():
            if not in_position:
                return False
            if position_side > 0:
                # Long stop: close drops below lower band - 1 ATR
                return low[i] < (lower_band[entry_bar] - 0.5 * atr[i])
            else:
                # Short stop: close rises above upper band + 1 ATR
                return high[i] > (upper_band[entry_bar] + 0.5 * atr[i])
        
        # === EXITS ===
        if in_position:
            stop_hit = check_stop()
            
            # Trend flip exit (trend changes AND min_hold)
            if position_side > 0 and st_bearish and min_hold:
                stop_hit = True
            if position_side < 0 and st_bullish and min_hold:
                stop_hit = True
            
            # Williams %R extreme reversal exit (optional profit taking)
            if position_side > 0 and is_overbought and (i - entry_bar) >= 4:
                stop_hit = True  # Take profit on overbought
            if position_side < 0 and is_oversold and (i - entry_bar) >= 4:
                stop_hit = True  # Take profit on oversold
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # ENTRY LOGIC:
            # Long when: Supertrend flips bullish + oversold + near 1d S3 + vol spike
            # OR: Supertrend bullish + very oversold + volume
            
            long_cond1 = (supertrend[i] == 1 and supertrend[i-1] == -1)  # ST cross up
            long_cond2 = is_oversold  # Williams %R oversold
            long_cond3 = near_1d_s3 or vol_spike[i]  # Near pivot or vol spike
            long_cond4 = uptrend  # 1d trend aligned
            
            if long_cond1 and long_cond2 and (long_cond3 or long_cond4):
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr[i]
                entry_bar = i
                signals[i] = SIZE
            
            # Short when: Supertrend flips bearish + overbought + near 1d R3 + vol spike
            short_cond1 = (supertrend[i] == -1 and supertrend[i-1] == 1)  # ST cross down
            short_cond2 = is_overbought  # Williams %R overbought
            short_cond3 = near_1d_r3 or vol_spike[i]  # Near pivot or vol spike
            short_cond4 = downtrend  # 1d trend aligned
            
            if short_cond1 and short_cond2 and (short_cond3 or short_cond4):
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr[i]
                entry_bar = i
                signals[i] = -SIZE
            
            # Fallback: Trend continuation entries if Williams %R confirms
            # Long continuation: ST already bullish + extreme oversold
            if st_bullish and is_oversold and vol_spike[i] and uptrend and not in_position:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr[i]
                entry_bar = i
                signals[i] = SIZE * 0.5  # Half size for continuation
            
            # Short continuation: ST already bearish + extreme overbought
            if st_bearish and is_overbought and vol_spike[i] and downtrend and not in_position:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr[i]
                entry_bar = i
                signals[i] = -SIZE * 0.5  # Half size for continuation
    
    return signals