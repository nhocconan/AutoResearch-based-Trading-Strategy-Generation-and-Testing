#!/usr/bin/env python3
"""
Experiment #007: 6h Weekly Pivot Mean Reversion + Extreme Volume + Regime

HYPOTHESIS: Weekly pivot levels represent major institutional S/R zones. When price 
deviates significantly from 6h mean (EMA21) AND touches weekly pivot AND shows 
extreme volume (>2.5x), institutions are defending levels → mean reversion play.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Mean reversion works in all regimes when price is at extreme deviations
- Weekly pivots are major levels that hold in bull AND bear markets
- Extreme volume confirms institutional defense of levels
- Regime filter (ADX) ensures we only trade when there's enough movement

KEY DIFFERENCE FROM FAILED STRATEGIES:
- Previous Camarilla (747 trades) was too loose → this uses WEEKLY pivots (higher TF)
- Volume threshold 2.5x (not 1.5x) → fewer but higher quality signals
- Mean reversion (not breakout) → different edge than Donchian failures
- Target: 75-150 total trades over 4 years (12-37/year)

ENTRY CONDITIONS (ALL must align):
1. Price within 1 ATR of weekly S3/R3 pivot
2. Price >2.5 ATR away from 6h EMA21 (extreme deviation)
3. Volume >2.5x 20-bar average (extreme spike)
4. ADX(14) > 20 (enough movement to profit)
5. 1d HMA trend confirmation (trade with HTF trend)

SIZE: 0.25 (discrete)
STOPLOSS: 2.5 ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weekly_pivot_meanrev_extreme_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(close, span):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=span, min_periods=span, adjust=False).mean().values

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
    
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

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

def calculate_weekly_pivots(prev_high, prev_low, prev_close):
    """
    Weekly pivot levels (standard floor pivots)
    PP = (H + L + C) / 3
    R1 = 2*PP - L
    R2 = PP + (H - L)
    R3 = H + 2*(PP - L)
    S1 = 2*PP - H
    S2 = PP - (H - L)
    S3 = L - 2*(H - PP)
    """
    n = len(prev_high)
    pivots = {
        'pp': np.full(n, np.nan, dtype=np.float64),
        'r1': np.full(n, np.nan, dtype=np.float64),
        'r2': np.full(n, np.nan, dtype=np.float64),
        'r3': np.full(n, np.nan, dtype=np.float64),
        's1': np.full(n, np.nan, dtype=np.float64),
        's2': np.full(n, np.nan, dtype=np.float64),
        's3': np.full(n, np.nan, dtype=np.float64),
    }
    
    for i in range(n):
        if np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]):
            continue
        
        h = prev_high[i]
        l = prev_low[i]
        c = prev_close[i]
        
        pp = (h + l + c) / 3.0
        pivots['pp'][i] = pp
        pivots['r1'][i] = 2.0 * pp - l
        pivots['r2'][i] = pp + (h - l)
        pivots['r3'][i] = h + 2.0 * (pp - l)
        pivots['s1'][i] = 2.0 * pp - h
        pivots['s2'][i] = pp - (h - l)
        pivots['s3'][i] = l - 2.0 * (h - pp)
    
    return pivots

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for trend bias
    def calc_hma(series, period):
        half = max(1, period // 2)
        sqrt_n = max(1, int(np.sqrt(period)))
        wma_half = series.ewm(span=half, min_periods=half, adjust=False).mean()
        wma_full = series.ewm(span=period, min_periods=period, adjust=False).mean()
        diff = 2.0 * wma_half - wma_full
        return diff.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean().values
    
    hma_1d_raw = calc_hma(pd.Series(df_1d['close'].values), 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate weekly pivots
    weekly_pivots = calculate_weekly_pivots(
        df_1w['high'].values,
        df_1w['low'].values,
        df_1w['close'].values
    )
    
    # Align weekly pivots to 6h
    s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivots['s3'])
    s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivots['s2'])
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivots['s1'])
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivots['r1'])
    r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivots['r2'])
    r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivots['r3'])
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    ema_21 = calculate_ema(close, 21)
    
    # Volume moving average
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK ===
        adx = adx_14[i]
        is_trending = adx > 20.0  # Need some directional movement
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        
        # === VOLUME CONFIRMATION (EXTREME) ===
        vol_extreme = vol_ratio[i] > 2.5  # Much stricter than 1.5x
        
        # === PRICE DEVIATION FROM MEAN ===
        if ema_21[i] > 1e-10:
            price_deviation = (close[i] - ema_21[i]) / atr_14[i]
        else:
            price_deviation = 0.0
        
        extreme_below = price_deviation < -2.5  # Price >2.5 ATR below EMA
        extreme_above = price_deviation > 2.5   # Price >2.5 ATR above EMA
        
        # === WEEKLY PIVOT PROXIMITY ===
        # Check if price is within 1 ATR of any major weekly pivot
        at_support = False
        at_resistance = False
        
        for pivot in [s1_aligned[i], s2_aligned[i], s3_aligned[i]]:
            if not np.isnan(pivot) and atr_14[i] > 0:
                dist = (close[i] - pivot) / atr_14[i]
                if -1.0 < dist < 1.5:  # Within 1 ATR below to 1.5 ATR above
                    at_support = True
                    break
        
        for pivot in [r1_aligned[i], r2_aligned[i], r3_aligned[i]]:
            if not np.isnan(pivot) and atr_14[i] > 0:
                dist = (pivot - close[i]) / atr_14[i]
                if -1.0 < dist < 1.5:
                    at_resistance = True
                    break
        
        # === ENTRY LOGIC (MEAN REVERSION) ===
        desired_signal = 0.0
        
        # LONG: Price at weekly support + extreme below EMA + volume + trend bias
        if is_trending and at_support and extreme_below and vol_extreme:
            if price_above_1d_hma:  # Only long if 1d trend is up
                desired_signal = SIZE
        
        # SHORT: Price at weekly resistance + extreme above EMA + volume + trend bias
        if is_trending and at_resistance and extreme_above and vol_extreme:
            if not price_above_1d_hma:  # Only short if 1d trend is down
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
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
        
        # === TAKE PROFIT (mean reversion to EMA) ===
        tp_triggered = False
        if in_position and position_side > 0:
            # TP when price returns to EMA (mean reversion complete)
            if close[i] >= ema_21[i]:
                tp_triggered = True
        
        if in_position and position_side < 0:
            if close[i] <= ema_21[i]:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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
        
        signals[i] = desired_signal
    
    return signals