#!/usr/bin/env python3
"""
Experiment #028: 12h Camarilla Pivot Breakout + Volume + Choppiness + 1w SMA Trend

HYPOTHESIS: Camarilla pivot levels (S3/S4/R3/R4) are mathematically derived from
previous day's range and represent institutional support/resistance zones.
When price breaks these levels with volume confirmation and the market is not
choppy (CHOP < 61.8), we have high-probability mean-reversion breakouts.

WHY 12h: Slower than 4h/6h = fewer but higher-quality signals. Institutional
players (prop desks, funds) operate on 4h-12h timeframes. 12h aligns with
their order flow. 54% keep rate for 12h strategies confirms this.

WHY IT WORKS IN BULL AND BEAR: Uses symmetrical pivot calculations. In bull,
we capture upside breakouts through R3/R4. In bear, downside breakdowns through
S3/S4. Choppiness filter keeps us out of whipsaw zones.

ENTRY CONDITIONS (must ALL agree):
1. Price breaks Camarilla level (R3/R4 for longs, S3/S4 for shorts)
2. Volume > 1.8x 20-bar average (strong institutional conviction)
3. 1w SMA200 confirms direction (bull filter for longs, bear filter for shorts)
4. Choppiness < 61.8 (market not ranging)

TARGET: 50-150 total trades over 4 years = 12-37/year.
Signal size: 0.25 (conservative, max 0.30).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_vol_chop_1w_v1"
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
    
    chop = pd.Series(chop).ffill().bfill().values
    return chop

def calculate_camarilla_levels(prev_high, prev_low, prev_close):
    """
    Calculate Camarilla pivot levels.
    R4 = close + (high - low) * 1.1/2
    R3 = close + (high - low) * 1.1/4
    R2 = close + (high - low) * 1.1/6
    R1 = close + (high - low) * 1.1/12
    S1 = close - (high - low) * 1.1/12
    S2 = close - (high - low) * 1.1/6
    S3 = close - (high - low) * 1.1/4
    S4 = close - (high - low) * 1.1/2
    """
    rng = prev_high - prev_low
    r4 = prev_close + rng * 0.55
    r3 = prev_close + rng * 0.275
    r2 = prev_close + rng * 0.1833
    r1 = prev_close + rng * 0.0917
    s1 = prev_close - rng * 0.0917
    s2 = prev_close - rng * 0.1833
    s3 = prev_close - rng * 0.275
    s4 = prev_close - rng * 0.55
    return r4, r3, r2, r1, s1, s2, s3, s4

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w SMA200 for trend (very long-term, won't change often)
    sma_200_1w = pd.Series(df_1w['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1w, sma_200_1w)
    
    # Local indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Camarilla levels (shift by 2 for previous day on 12h = 2 bars ago)
    r4 = np.zeros(n)
    r3 = np.zeros(n)
    r2 = np.zeros(n)
    r1 = np.zeros(n)
    s1 = np.zeros(n)
    s2 = np.zeros(n)
    s3 = np.zeros(n)
    s4 = np.zeros(n)
    
    for i in range(4, n):  # Need at least 2 bars back for previous day
        prev_high_i = high[i - 2]  # Previous 12h bar (1 day ago)
        prev_low_i = low[i - 2]
        prev_close_i = close[i - 2]
        r4_i, r3_i, r2_i, r1_i, s1_i, s2_i, s3_i, s4_i = calculate_camarilla_levels(
            prev_high_i, prev_low_i, prev_close_i)
        r4[i] = r4_i
        r3[i] = r3_i
        r2[i] = r2_i
        r1[i] = r1_i
        s1[i] = s1_i
        s2[i] = s2_i
        s3[i] = s3_i
        s4[i] = s4_i
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 220  # Need 200 for 1w SMA200 + buffer
    
    for i in range(warmup, n):
        # Validation checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_200_aligned[i]):
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
        
        current_close = close[i]
        current_high = high[i]
        current_low = low[i]
        prev_close = close[i - 1] if i > 0 else current_close
        
        # === REGIME CHECK ===
        # Only trade when not too choppy
        is_choppy = chop[i] > 61.8
        
        if is_choppy and not in_position:
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION (1w SMA200) ===
        price_above_1w_sma = current_close > sma_200_aligned[i]
        price_below_1w_sma = current_close < sma_200_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.8  # Strict: 1.8x average
        
        # === CAMARILLA LEVELS ===
        prev_r3 = r3[i - 1] if i > 0 else r3[i]
        prev_r4 = r4[i - 1] if i > 0 else r4[i]
        prev_s3 = s3[i - 1] if i > 0 else s3[i]
        prev_s4 = s4[i - 1] if i > 0 else s4[i]
        
        # Detect breakout: price closes beyond previous level
        breakout_above_r3 = prev_close < prev_r3 and current_close > prev_r3
        breakout_above_r4 = prev_close < prev_r4 and current_close > prev_r4
        breakout_below_s3 = prev_close > prev_s3 and current_close < prev_s3
        breakout_below_s4 = prev_close > prev_s4 and current_close < prev_s4
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            # === LONG: Price breaks above R3 with volume + trend confirmation ===
            if (breakout_above_r3 or breakout_above_r4) and price_above_1w_sma:
                if vol_spike:
                    desired_signal = SIZE
            
            # === SHORT: Price breaks below S3 with volume + trend confirmation ===
            if (breakout_below_s3 or breakout_below_s4) and price_below_1w_sma:
                if vol_spike:
                    desired_signal = -SIZE
        
        # === STOPLOSS (3.0 ATR - wider for breakout trades) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT at 2.5R ===
        if in_position and position_side > 0:
            profit_target = entry_price + 2.5 * entry_atr
            if high[i] >= profit_target:
                desired_signal = SIZE / 2  # Half position
        
        if in_position and position_side < 0:
            profit_target = entry_price - 2.5 * entry_atr
            if low[i] <= profit_target:
                desired_signal = -SIZE / 2  # Half position
        
        # === TIME EXIT (hold 4 bars = 2 days max for Camarilla trades) ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 4:
            # Exit on momentum reversal
            if position_side > 0 and current_close < s2[i]:
                desired_signal = 0.0
            if position_side < 0 and current_close > r2[i]:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = current_close
                entry_atr = atr_14[i]
                highest_since_entry = current_high
                lowest_since_entry = current_low
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
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