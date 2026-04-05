#!/usr/bin/env python3
"""
Experiment #7851: 6-hour Camarilla Pivot + Volume Spike + Regime Filter
Hypothesis: On 6B timeframe, price rejection at Camarilla R3/S3 levels with volume >2x 20-period mean and ADX < 25 (range regime) captures mean reversion in both bull and bear markets. Breakouts at R4/S4 with volume confirmation continue the trend. Uses 1d trend filter (price vs EMA50) for directional bias. Targets 50-150 trades over 4 years with controlled risk via ATR stops.
"""

from mtf_data import get_aff_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7851_6h_camarilla_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1 / 12  # Standard Camarilla multiplier
PIVOT_LOOKBACK = 1  # Use previous day for pivots
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
ADX_PERIOD = 14
ADX_RANGE_THRESHOLD = 25  # ADX < 25 = range (mean revert)
EMA_TREND_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_adx(high, low, close, period):
    """Calculate ADX with proper smoothing"""
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), 
                    np.abs(low - np.roll(close, 1)))
    
    # Wilder's smoothing
    atr = np.zeros_like(tr)
    atr[0] = np.nan
    for i in range(1, len(tr)):
        if np.isnan(atr[i-1]):
            atr[i] = np.nanmean(tr[max(0, i-period+1):i+1])
        else:
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = 100 * (np.zeros_like(plus_dm))
    minus_di = 100 * (np.zeros_like(minus_dm))
    
    # Smooth DM
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    for i in range(len(plus_dm)):
        if i < period:
            plus_dm_smooth[i] = np.nan
            minus_dm_smooth[i] = np.nan
        else:
            if i == period:
                plus_dm_smooth[i] = np.nansum(plus_dm[max(0, i-period+1):i+1])
                minus_dm_smooth[i] = np.nansum(minus_dm[max(0, i-period+1):i+1])
            else:
                plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1]/period) + plus_dm[i]
                minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1]/period) + minus_dm[i]
    
    # Avoid division by zero
    divisor = plus_dm_smooth + minus_dm_smooth
    plus_di = np.where(divisor != 0, 100 * plus_dm_smooth / divisor, 0)
    minus_di = np.where(divisor != 0, 100 * minus_dm_smooth / divisor, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # Smooth DX to get ADX
    adx = np.zeros_like(dx)
    for i in range(len(dx)):
        if i < period:
            adx[i] = np.nan
        elif i == period:
            adx[i] = np.nanmean(dx[max(0, i-period+1):i+1])
        else:
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    
    # Trend bias: above EMA = bullish, below EMA = bearish
    trend_bias_1d = np.where(close_1d > ema_1d, 1, -1)  # 1=bullish, -1=bearish
    trend_bias_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    # Calculate daily pivot points (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Camarilla levels
    range_1d = high_1d - low_1d
    camarilla_r3 = pivot + (range_1d * CAMARILLA_MULT * 3)
    camarilla_s3 = pivot - (range_1d * CAMARILLA_MULT * 3)
    camarilla_r4 = pivot + (range_1d * CAMARILLA_MULT * 4)
    camarilla_s4 = pivot - (range_1d * CAMARILLA_MULT * 4)
    
    # Align Camarilla levels to 6t
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ADX for regime detection
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # ATR for risk management
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ADX_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(trend_bias_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market conditions
        bull_bias = trend_bias_1d_aligned[i] == 1   # 1d close above EMA
        bear_bias = trend_bias_1d_aligned[i] == -1  # 1d close below EMA
        range_regime = adx[i] < ADX_RANGE_THRESHOLD  # ADX < 25 = range
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Mean reversion at R3/S3 (fade) in range regime
        fade_r3 = close[i] <= camarilla_r3_aligned[i] and camarilla_r3_aligned[i] > 0
        fade_s3 = close[i] >= camarilla_s3_aligned[i] and camarilla_s3_aligned[i] > 0
        
        # Breakout continuation at R4/S4
        breakout_r4 = close[i] >= camarilla_r4_aligned[i] and camarilla_r4_aligned[i] > 0
        breakout_s4 = close[i] <= camarilla_s4_aligned[i] and camarilla_s4_aligned[i] > 0
        
        # Entry logic
        long_entry = False
        short_entry = False
        
        if range_regime and volume_confirmed:
            # In range: fade at R3/S3
            if bull_bias and fade_s3:  # Bullish bias, price at S3 -> long
                long_entry = True
            elif bear_bias and fade_r3:  # Bearish bias, price at R3 -> short
                short_entry = True
        else:
            # Trending: breakout continuation at R4/S4
            if bull_bias and breakout_r4:  # Bullish bias, break above R4 -> long
                long_entry = True
            elif bear_bias and breakout_s4:  # Bearish bias, break below S4 -> short
                short_entry = True
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals