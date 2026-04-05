#!/usr/bin/env python3
"""
Experiment #8115: 6-hour Weekly Pivot + Volume Breakout with Daily Trend Filter
Hypothesis: Weekly pivot levels provide strong support/resistance zones. 
Breaking above weekly R3 or below S3 with volume > 1.5x 20-period MA and 
aligned with daily trend (price above/below daily EMA50) captures significant moves.
Weekly timeframe reduces noise, daily provides trend context, 6h offers balance 
between signal frequency and cost efficiency. Targets 12-37 trades/year per symbol.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8115_6h_weekly_pivot_daily_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
WEEKLY_LOOKBACK = 1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
EMA_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points and support/resistance levels"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points (using previous week's data)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot points for each week
    wpivot = np.full_like(weekly_close, np.nan)
    wr1 = np.full_like(weekly_close, np.nan)
    wr2 = np.full_like(weekly_close, np.nan)
    wr3 = np.full_like(weekly_close, np.nan)
    ws1 = np.full_like(weekly_close, np.nan)
    ws2 = np.full_like(weekly_close, np.nan)
    ws3 = np.full_like(weekly_close, np.nan)
    
    for i in range(len(weekly_close)):
        if not (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or np.isnan(weekly_close[i])):
            p, r1, r2, r3, s1, s2, s3 = calculate_pivot_points(weekly_high[i], weekly_low[i], weekly_close[i])
            wpivot[i] = p
            wr1[i] = r1
            wr2[i] = r2
            wr3[i] = r3
            ws1[i] = s1
            ws2[i] = s2
            ws3[i] = s3
    
    # Calculate daily EMA for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    price_vs_ema = np.where(daily_close > daily_ema, 1, -1)  # 1=bullish, -1=bearish
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Align HTF data to LTF
    wr3_aligned = align_htf_to_ltf(prices, df_1w, wr3)
    ws3_aligned = align_htf_to_ltf(prices, df_1w, ws3)
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD, EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(wr3_aligned[i]) or np.isnan(ws3_aligned[i]) or np.isnan(price_vs_ema_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from daily EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # daily close above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # daily close below EMA50
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions - weekly pivot levels
        weekly_breakout_up = close[i] > wr3_aligned[i-1] if i-1 >= 0 and not np.isnan(wr3_aligned[i-1]) else False
        weekly_breakout_down = close[i] < ws3_aligned[i-1] if i-1 >= 0 and not np.isnan(ws3_aligned[i-1]) else False
        
        # Entry conditions
        long_entry = bull_bias and weekly_breakout_up and volume_confirmed
        short_entry = bear_bias and weekly_breakout_down and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals