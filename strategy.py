#!/usr/bin/env python3
"""
Experiment #028: 4h Tight Camarilla + Volume Spike + Choppiness + Trend Bias

HYPOTHESIS: Camarilla pivot levels are mathematically precise support/resistance
where institutional order flow clusters. Using tighter S4/R4 levels (stronger 
breakdown/reversal zones) with volume spike confirmation and Choppiness Index
filtering creates high-probability mean reversion setups.

The 1d SMA200 adds trend bias: longs only when price above SMA200 (bull bounce),
shorts only when price below SMA200 (bear rally fade). This reduces whipsaws
in choppy markets.

WHY IT WORKS IN BULL AND BEAR: Camarilla is symmetric - both sides work.
In bull: price bounces from S3/S4 (bull market corrections). 
In bear: price reverses from R3/R4 (bear market rallies).
Volume spike confirms institutional conviction. Choppiness keeps us out of 
trend-following whipsaws.

TARGET: 75-150 total trades over 4 years (19-37/year). HARD MAX: 300.
Signal size: 0.30 (moderate).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_vol_chop_trend_v1"
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

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels
    S3/S4 = support (buy zone), R3/R4 = resistance (sell zone)
    """
    pivot = (high + low + close) / 3.0
    h4 = close + (high - low) * 1.1 / 2.0
    h3 = close + (high - low) * 1.1 / 4.0
    h2 = close + (high - low) * 1.1 / 6.0
    h1 = close + (high - low) * 1.1 / 12.0
    l1 = close - (high - low) * 1.1 / 12.0
    l2 = close - (high - low) * 1.1 / 6.0
    l3 = close - (high - low) * 1.1 / 4.0
    l4 = close - (high - low) * 1.1 / 2.0
    return {'h4': h4, 'h3': h3, 'h2': h2, 'h1': h1, 
            'l1': l1, 'l2': l2, 'l3': l3, 'l4': l4, 'pivot': pivot}

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction
    sma_200 = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
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
        
        # === TREND DIRECTION (1d SMA200) ===
        price_above_1d_sma = close[i] > sma_200_aligned[i]
        
        # === REGIME (Choppiness Index) ===
        # Skip if too choppy (avoid whipsaws)
        is_choppy = chop[i] > 61.8
        
        if is_choppy and not in_position:
            signals[i] = 0.0
            continue
        
        # === Camarilla levels ===
        camarilla = calculate_camarilla(high[i], low[i], close[i])
        s3 = camarilla['s3']
        s4 = camarilla['s4']
        r3 = camarilla['r3']
        r4 = camarilla['r4']
        pivot = camarilla['pivot']
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Price bounces from S3/S4 with volume ===
            # Low touches or接近 S3 or S4 zone
            low_touch = low[i] <= s3 * 1.002  # Allow 0.2% buffer
            
            if low_touch and price_above_1d_sma and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: Price reverses from R3/R4 with volume ===
            # High touches or接近 R3 or R4 zone
            high_touch = high[i] >= r3 * 0.998  # Allow 0.2% buffer
            
            if high_touch and not price_above_1d_sma and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT: Trail to breakeven after 1.5R ===
        if in_position and position_side > 0:
            profit_r = (close[i] - entry_price) / entry_atr
            if profit_r > 1.5:
                stop_price = max(stop_price, entry_price + 0.2 * entry_atr)
        
        if in_position and position_side < 0:
            profit_r = (entry_price - close[i]) / entry_atr
            if profit_r > 1.5:
                stop_price = min(stop_price, entry_price - 0.2 * entry_atr)
        
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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