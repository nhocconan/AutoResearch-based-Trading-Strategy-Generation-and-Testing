#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index regime filter with 1-day EMA trend filter and 12h momentum
# Uses 1d EMA34 for trend filter, 12h Choppiness Index for regime detection, and 12h RSI for momentum
# Works in bull/bear via trend filter and regime filter - mean reversion in range (CHOP>61.8), trend following in trend (CHOP<38.2)
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "12h_Choppiness_Index_EMA34_RSI_Momentum"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_daily = df_daily['close'].values
    ema_daily_34 = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 12h True Range for Choppiness Index
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate 12h ATR(14) for Choppiness Index denominator
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h highest high and lowest low over 14 periods for Choppiness Index numerator
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = highest_high_14 - lowest_low_14
    
    # Calculate 12h Choppiness Index (CHOP)
    # CHOP = 100 * log10(sum(ATR(14)) / (HHV(14) - LLV(14))) / log10(14)
    # Avoid division by zero and log of zero
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr_14 / np.maximum(range_14, 1e-10)) / np.log10(14)
    chop = np.where(np.isfinite(chop), chop, 50)  # Replace inf/NaN with neutral value
    
    # Calculate 12h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align daily EMA to 12h timeframe
    ema_daily_34_aligned = align_htf_to_ltf(prices, df_daily, ema_daily_34)
    
    # Pre-compute session filter (08-20 UTC) - though less critical for 12h
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session (optional for 12h)
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if (np.isnan(ema_daily_34_aligned[i]) or np.isnan(chop[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: daily EMA34 direction (using previous bar to avoid look-ahead)
        ema_now = ema_daily_34_aligned[i]
        ema_prev = ema_daily_34_aligned[i-1]
        ema_uptrend = ema_now > ema_prev
        ema_downtrend = ema_now < ema_prev
        
        # Regime filters based on Choppiness Index
        chop_value = chop[i]
        is_range = chop_value > 61.8  # Range bound - mean revert
        is_trend = chop_value < 38.2  # Trending - trend follow
        
        # Momentum filter: RSI levels
        rsi_value = rsi[i]
        rsi_oversold = rsi_value < 30
        rsi_overbought = rsi_value > 70
        
        if position == 0:
            # Look for entries based on regime
            if is_range and rsi_oversold and ema_uptrend:
                # In range, oversold with bullish bias - go long
                signals[i] = 0.25
                position = 1
            elif is_range and rsi_overbought and ema_downtrend:
                # In range, overbought with bearish bias - go short
                signals[i] = -0.25
                position = -1
            elif is_trend and rsi_value > 50 and ema_uptrend:
                # In trend, bullish momentum - go long
                signals[i] = 0.25
                position = 1
            elif is_trend and rsi_value < 50 and ema_downtrend:
                # In trend, bearish momentum - go short
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: range extreme or trend/momentum reversal
            if (is_range and rsi_value > 70) or (not ema_uptrend) or (is_trend and rsi_value < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: range extreme or trend/momentum reversal
            if (is_range and rsi_value < 30) or (not ema_downtrend) or (is_trend and rsi_value > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals