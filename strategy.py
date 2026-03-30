#!/usr/bin/env python3
"""
Experiment #008: 12h Camarilla S4/R4 Mean-Reversion with 1w Trend + Volume

HYPOTHESIS: After extended moves, price snaps back to Camarilla S4 (support)
or R4 (resistance) levels. These are tighter bands than S3/R3, catching
smaller reversals. Combined with 1w trend alignment (prevents fighting
major trends) and volume confirmation, this captures 1-3 day reversals
in both bull and bear markets.

WHY 12h + 1w: 12h captures multi-day swings. 1w trend alignment prevents
entering against major bull/bear cycles. Symmetric entry logic works in
both directions.

TARGET: 75-200 total trades over 4 years. Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_s4r4_1w_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper ATR calculation"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w SMA200 for major trend (prevents fighting multi-month trends)
    sma_200_1w = pd.Series(df_1w['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1w, sma_200_1w)
    
    # 1w EMA21 for faster trend confirmation
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume rolling average (20 periods = 10 days)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, 1)
    
    # 12h EMA for short-term trend
    ema_12h = pd.Series(close).ewm(span=8, min_periods=8, adjust=False).mean().values
    
    # Signals
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
    entry_ema = 0.0
    
    warmup = max(200, 50)  # Need enough for 1w SMA200 alignment buffer
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if 1w indicators not aligned
        if np.isnan(sma_200_aligned[i]) or np.isnan(ema_21_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1w) ===
        # Long only when price above 1w SMA200 and EMA21 > SMA200 (confirmed uptrend)
        # Short only when price below 1w SMA200 and EMA21 < SMA200 (confirmed downtrend)
        bull_market = close[i] > sma_200_aligned[i] and ema_21_aligned[i] > sma_200_aligned[i]
        bear_market = close[i] < sma_200_aligned[i] and ema_21_aligned[i] < sma_200_aligned[i]
        
        # Short-term momentum: 12h EMA direction
        short_bull = close[i] > ema_12h[i]
        short_bear = close[i] < ema_12h[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === CAMARILLA LEVELS from previous CLOSED bar ===
        prev_high = high[i - 1]
        prev_low = low[i - 1]
        prev_close = close[i - 1]
        prev_open = prev_close - (close[i - 1] - low[i - 1])  # rough open estimate
        prev_range = prev_high - prev_low
        
        # Classic Camarilla S4/R4 levels
        # R4 = Close + Range * 0.18333 (0.11/60 roughly)
        # S4 = Close - Range * 0.18333
        r4 = prev_close + prev_range * 0.18333
        s4 = prev_close - prev_range * 0.18333
        
        # Additional reference: mid-point
        mid = (prev_high + prev_low) / 2.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Price drops to S4 with trend alignment + volume ===
            # Only long in bull markets or during bear market rallies
            if (bull_market or not bear_market) and vol_spike:
                # Price must be within 0.3 ATR of S4 (not already bounced past)
                if low[i] <= s4 and low[i] >= s4 - 0.5 * atr_14[i]:
                    desired_signal = SIZE
                    # Also check we're not too extended from the level
        
            # === SHORT: Price rallies to R4 with trend alignment + volume ===
            if (bear_market or not bull_market) and vol_spike:
                # Price must be within 0.3 ATR of R4
                if high[i] >= r4 and high[i] <= r4 + 0.5 * atr_14[i]:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR — give trades room to breathe) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TAKE PROFIT (3R or ATR trailing stop tightens) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 2:
            # Target 3x risk
            if position_side > 0:
                profit_target = entry_price + 3.0 * entry_atr
                if close[i] >= profit_target:
                    desired_signal = 0.0
                # Also exit if short-term trend flips against us
                if short_bear and bars_held >= 3:
                    desired_signal = 0.0
            
            if position_side < 0:
                profit_target = entry_price - 3.0 * entry_atr
                if close[i] <= profit_target:
                    desired_signal = 0.0
                # Also exit if short-term trend flips against us
                if short_bull and bars_held >= 3:
                    desired_signal = 0.0
        
        # === UPDATE POSITION ===
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
                entry_ema = ema_12h[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals