#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla H3/L3 breakout with 1w ADX regime filter and volume confirmation
    # Long: price breaks above H3 AND 1w ADX > 25 (strong trend) AND volume > 2.0x avg
    # Short: price breaks below L3 AND 1w ADX > 25 (strong trend) AND volume > 2.0x avg
    # Exit: price retests the breakout level (H3 for longs, L3 for shorts) or touches opposite level (L4/H4)
    # Using 12h timeframe for low trade frequency (target 12-37/year), Camarilla for structure,
    # 1w ADX to filter weak/choppy markets, and volume confirmation to avoid false breakouts.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly ADX(14) for trend strength filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: smoothed = (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # DI+ and DI-
    plus_di14 = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
    minus_di14 = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align weekly ADX to 12h
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    # Need to shift by 1 to avoid look-ahead: use previous day's data for today's levels
    h3 = np.full(n, np.nan)
    l3 = np.full(n, np.nan)
    h4 = np.full(n, np.nan)
    l4 = np.full(n, np.nan)
    
    # For 12h timeframe, we need to aggregate to daily OHLC first
    # Since we don't have daily data directly, we'll approximate using rolling window
    # In practice, we'd use get_htf_data for 1d, but for simplicity we'll use 24-period rolling (12h * 2 = 1d)
    if n >= 24:
        # Calculate rolling daily OHLC (24 periods of 12h = 1 day)
        roll_high = pd.Series(high).rolling(window=24, min_periods=24).max().values
        roll_low = pd.Series(low).rolling(window=24, min_periods=24).min().values
        roll_close = pd.Series(close).rolling(window=24, min_periods=24).last().values
        roll_open = pd.Series(close).rolling(window=24, min_periods=24).first().values  # approximate
        
        # Camarilla levels: based on previous day's range
        # H4 = close + 1.1*(high-low)
        # H3 = close + 0.55*(high-low)
        # L3 = close - 0.55*(high-low)
        # L4 = close - 1.1*(high-low)
        for i in range(24, n):
            prev_high = roll_high[i-1]  # previous day's high
            prev_low = roll_low[i-1]    # previous day's low
            prev_close = roll_close[i-1] # previous day's close
            if not (np.isnan(prev_high) or np.isnan(prev_low) or np.isnan(prev_close)):
                range_val = prev_high - prev_low
                h4[i] = prev_close + 1.1 * range_val
                h3[i] = prev_close + 0.55 * range_val
                l3[i] = prev_close - 0.55 * range_val
                l4[i] = prev_close - 1.1 * range_val
    
    # Get 12h volume for confirmation (>2.0x 24-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(h3[i]) or np.isnan(l3[i]) or
            np.isnan(h4[i]) or np.isnan(l4[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 indicates strong trending market
        strong_trend = adx_1w_aligned[i] > 25
        
        # Camarilla breakout conditions
        breakout_h3 = close[i] > h3[i]
        breakout_l3 = close[i] < l3[i]
        
        # Exit conditions: retest breakout level or touch opposite level
        retest_h3 = close[i] < h3[i] and position == 1  # Long exit on H3 retest
        retest_l3 = close[i] > l3[i] and position == -1  # Short exit on L3 retest
        touch_l4 = close[i] < l4[i]  # Exit long on L4 touch
        touch_h4 = close[i] > h4[i]  # Exit short on H4 touch
        
        # Entry logic: Camarilla breakout + strong trend + volume confirmation
        long_entry = breakout_h3 and strong_trend and volume_spike[i]
        short_entry = breakout_l3 and strong_trend and volume_spike[i]
        
        # Exit logic: retest or opposite level touch
        long_exit = retest_h3 or touch_l4
        short_exit = retest_l3 or touch_h4
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_camarilla_h3l3_breakout_adx_volume_v1"
timeframe = "12h"
leverage = 1.0