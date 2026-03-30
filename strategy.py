#!/usr/bin/env python3
"""
Experiment #028: 4h Camarilla Pivot + 12h SMA200 Trend + Volume + Choppiness

HYPOTHESIS: Camarilla pivot levels (S3/R3) are proven support/resistance where
price reversals frequently occur. By combining with 12h SMA200 for trend direction
and Choppiness to filter ranging markets, this captures mean-reversion trades
at key structural levels with institutional confirmation.

WHY IT WORKS IN BULL AND BEAR:
- Bull: Buy at S3 pullbacks (discount entry), ride to R3
- Bear: Sell at R3 rallies (premium entry), ride to S3
- Symmetrical levels = works both directions
- Volume confirms institutional participation at levels
- Choppiness keeps us out of trendless markets

KEY INSIGHT FROM DB: gen_camarilla_pivot_volume_spike_choppiness_4h_v1 achieved
test Sharpe=1.471 with 95 trades. This strategy is the DIRECT inspiration.

TARGET: 75-150 total trades over 4 years (19-37/year). HARD MAX: 300.
Signal size: 0.25 discrete levels.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_vol_chop_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_camarilla_levels(high, low, close):
    """
    Camarilla pivot levels.
    S3/S4 = support levels (buy zones)
    R3/R4 = resistance levels (sell zones)
    """
    n = len(close)
    pivot = (high + low + close) / 3.0
    rng = high - low
    
    # Classic Camarilla multipliers
    r4 = close + rng * 0.55
    r3 = close + rng * 0.275
    r2 = close + rng * 0.183
    r1 = close + rng * 0.0916
    
    s1 = close - rng * 0.0916
    s2 = close - rng * 0.183
    s3 = close - rng * 0.275
    s4 = close - rng * 0.55
    
    return r4, r3, r2, r1, pivot, s1, s2, s3, s4

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h SMA200 for trend direction
    sma_200_12h = pd.Series(df_12h['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_12h, sma_200_12h)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Camarilla levels (calculated on previous bar to avoid look-ahead)
    r4, r3, r2, r1, pivot, s1, s2, s3, s4 = calculate_camarilla_levels(high, low, close)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    entry_bar = 0
    
    warmup = 250  # Need enough for SMA200(12h) + Camarilla + buffer
    
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
        
        # === TREND DIRECTION (12h SMA200) ===
        # Trend up: price above SMA200
        # Trend down: price below SMA200
        price_above_12h_sma = close[i] > sma_200_aligned[i]
        trend_up = price_above_12h_sma
        
        # === REGIME (Choppiness Index) ===
        # Only trade in trending or neutral markets (CHOP < 61.8)
        # Skip when too choppy (no clear direction)
        is_choppy = chop[i] > 61.8
        
        if is_choppy and not in_position:
            signals[i] = 0.0
            continue
        
        # === VOLUME CONFIRMATION ===
        # Require volume spike at level touch
        vol_spike = vol_ratio[i] > 1.3
        
        # Previous bar's Camarilla levels (shifted by 1)
        prev_s3 = s3[i - 1] if i > 0 else 0
        prev_s4 = s4[i - 1] if i > 0 else 0
        prev_r3 = r3[i - 1] if i > 0 else 0
        prev_r4 = r4[i - 1] if i > 0 else 0
        
        # Current price at levels?
        current_high = high[i]
        current_low = low[i]
        
        # Check if price touched S3 (potential long entry)
        touched_s3 = current_low <= prev_s3
        # Check if price touched R3 (potential short entry)
        touched_r3 = current_high >= prev_r3
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            # === LONG ENTRY: Price touches S3 support + trend up ===
            # S3 is strong support. If price bounces from S3 with volume, go long.
            # Target: R3 (resistance) or higher
            if touched_s3 and trend_up:
                if vol_spike:  # Volume confirmation
                    desired_signal = SIZE
            
            # === SHORT ENTRY: Price touches R3 resistance + trend down ===
            # R3 is strong resistance. If price rejects from R3 with volume, go short.
            # Target: S3 (support) or lower
            if touched_r3 and not trend_up:
                if vol_spike:  # Volume confirmation
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Long stoploss: below entry - 2.5*ATR OR below S4
            stoploss_price = min(entry_price - 2.5 * entry_atr, prev_s4)
            if low[i] < stoploss_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            # Short stoploss: above entry + 2.5*ATR OR above R4
            stoploss_price = max(entry_price + 2.5 * entry_atr, prev_r4)
            if high[i] > stoploss_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT / TIME EXIT ===
        bars_held = i - entry_bar
        
        if in_position:
            # Take profit at 2:1 R:R OR after 8 bars (2 days on 4h)
            r_multiple = 0.0
            if position_side > 0:
                r_multiple = (close[i] - entry_price) / entry_atr
            else:
                r_multiple = (entry_price - close[i]) / entry_atr
            
            # TP at 2R or time exit
            if r_multiple >= 2.0 or bars_held >= 8:
                desired_signal = 0.0
        
        # === TREND CHANGE EXIT ===
        # If trend changes, close position
        if in_position:
            if position_side > 0 and not trend_up:
                # Was long but trend turned down
                desired_signal = 0.0
            if position_side < 0 and trend_up:
                # Was short but trend turned up
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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
        
        signals[i] = desired_signal
    
    return signals