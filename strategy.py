#!/usr/bin/env python3
"""
Experiment #028: 12h Camarilla Pivot + Volume Spike + 1w SMA200 Trend

HYPOTHESIS: Camarilla pivot levels are mathematically derived support/resistance
that institutional traders use. Price frequently bounces off S2/S3 supports and
R2/R3 resistances. By combining these with:
1. Volume spike confirmation (institutional participation)
2. 1w SMA200 trend filter (avoid countertrend trades)
3. Choppiness regime filter (avoid ranging markets)

This creates VERY selective entries (~25-50 trades over 4 years) that should
have high win rate. The tight entry conditions prevent overtrading.

WHY 12h: Slow enough to avoid fee drag (0.10% RT), fast enough to capture
medium-term swings. Matches the recommended timeframe from experiment guidelines.

WHY IT WORKS IN BULL AND BEAR:
- Bull: Price breaks up through Camarilla resistance with volume + trend up = LONG
- Bear: Price fails at Camarilla resistance or bounces off S2/S3 with trend down = SHORT
- Range: Choppiness filter keeps us flat

TARGET: 25-50 total trades over 4 years (6-12/year). HARD MAX: 100.
Previous similar strategy (ETH, 95 trades) achieved test Sharpe=1.47.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_vol_1w_trend_v1"
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

def calculate_camarilla_levels(high, low, close):
    """
    Calculate Camarilla pivot levels.
    Using classic Camarilla formula with 4 levels.
    S3/S4 = support, R3/R4 = resistance
    """
    n = len(close)
    h_range = high - low
    
    pivot = (high + low + close) / 3.0
    
    s1 = close + h_range * 0.09167
    s2 = close + h_range * 0.18333
    s3 = close + h_range * 0.27500
    s4 = close + h_range * 0.36667
    
    r1 = close - h_range * 0.09167
    r2 = close - h_range * 0.18333
    r3 = close - h_range * 0.27500
    r4 = close - h_range * 0.36667
    
    return {
        'pivot': pivot,
        's1': s1, 's2': s2, 's3': s3, 's4': s4,
        'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4
    }

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
    
    # === Load 1w HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w SMA200 for trend direction (slowest filter = most selective)
    sma_1w = pd.Series(df_1w['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # Local indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    camarilla = calculate_camarilla_levels(high, low, close)
    
    # Volume 20-bar MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    # Warmup: need 20 bars for Camarilla + volume MA + chop + buffer
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Current Camarilla levels
        s1 = camarilla['s1'][i]
        s2 = camarilla['s2'][i]
        s3 = camarilla['s3'][i]
        r1 = camarilla['r1'][i]
        r2 = camarilla['r2'][i]
        r3 = camarilla['r3'][i]
        
        # Previous Camarilla levels for touch detection
        s1_prev = camarilla['s1'][i-1]
        s2_prev = camarilla['s2'][i-1]
        s3_prev = camarilla['s3'][i-1]
        r1_prev = camarilla['r1'][i-1]
        r2_prev = camarilla['r2'][i-1]
        r3_prev = camarilla['r3'][i-1]
        
        # === TREND FILTER (1w SMA200) ===
        price_above_1w_sma = close[i] > sma_1w_aligned[i]
        
        # === REGIME FILTER (Choppiness) ===
        # Only trade when not too choppy (CHOP < 61.8)
        is_choppy = chop[i] > 61.8
        
        if is_choppy and not in_position:
            signals[i] = 0.0
            continue
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.8  # 80% above average
        
        # === PRICE TOUCH DETECTION ===
        # Check if price touched (or bounced from) Camarilla levels
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        current_close = close[i]
        
        # Long: price bounces from S2 or S3 support
        # Bounce = price was near/below level, now recovering
        touched_s2 = (prev_low <= s2_prev * 1.002) and (current_close > s2 * 0.998)
        touched_s3 = (prev_low <= s3_prev * 1.002) and (current_close > s3 * 0.998)
        
        # Short: price rejected at R2 or R3 resistance
        # Rejection = price was near/above level, now falling
        touched_r2 = (prev_high >= r2_prev * 0.998) and (current_close < r2 * 1.002)
        touched_r3 = (prev_high >= r3_prev * 0.998) and (current_close < r3 * 1.002)
        
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Price bounced from S2 or S3 + volume spike + trend up + not choppy
            if (touched_s2 or touched_s3) and vol_spike and price_above_1w_sma:
                desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Price rejected at R2 or R3 + volume spike + trend down
            if (touched_r2 or touched_r3) and vol_spike and not price_above_1w_sma:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            trailing_stop = prev_close - 2.5 * entry_atr
            if current_close < trailing_stop:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            trailing_stop = prev_close + 2.5 * entry_atr
            if current_close > trailing_stop:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === MINIMUM HOLDING PERIOD (4 bars = 2 days) ===
        bars_held = i - entry_bar
        if bars_held < 4:
            # Keep position regardless of other signals
            if in_position:
                desired_signal = position_side * SIZE
        
        # === TAKE PROFIT (3R) ===
        if in_position and bars_held >= 4:
            if position_side > 0:
                profit_r = (current_close - entry_price) / entry_atr
                if profit_r >= 3.0:
                    # Take partial profit, reduce to half position
                    if desired_signal == SIZE:
                        desired_signal = SIZE / 2
            elif position_side < 0:
                profit_r = (entry_price - current_close) / entry_atr
                if profit_r >= 3.0:
                    if desired_signal == -SIZE:
                        desired_signal = -SIZE / 2
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = current_close
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
        
        signals[i] = desired_signal
    
    return signals