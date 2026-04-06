#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with 12h trend filter and volume confirmation.
# In ranging markets, price often reverses at Camarilla R3/S3 levels.
# In trending markets, breakouts at R4/S4 levels with volume continue the trend.
# Use 12h EMA(50) slope for trend filter: only take R3/S3 reversals in range, R4/S4 breakouts in trend.
# Works in both bull and bear markets as it adapts to regime.
# Target: 12-37 trades/year by using strict pivot levels + trend + volume.

name = "exp_13619_6h_camarilla_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 10  # lookback for high/low
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the period"""
    # Using typical Camarilla formula based on previous period's range
    # H = high, L = low, C = close of lookback period
    H = np.max(high)
    L = np.min(low)
    C = close[-1]  # current close as reference
    RANGE = H - L
    
    # Camarilla levels
    R4 = C + (RANGE * 1.500)
    R3 = C + (RANGE * 1.250)
    S3 = C - (RANGE * 1.250)
    S4 = C - (RANGE * 1.500)
    
    return R4, R3, S3, S4

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter and Camarilla calculation ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, TREND_EMA_PERIOD)
    ema_12h_slope = np.diff(ema_12h, prepend=ema_12h[0])  # slope approximation
    ema_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_slope)
    
    # Calculate 12h Camarilla levels for each bar (using lookback window)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    camarilla_R4 = np.full(len(close_12h), np.nan)
    camarilla_R3 = np.full(len(close_12h), np.nan)
    camarilla_S3 = np.full(len(close_12h), np.nan)
    camarilla_S4 = np.full(len(close_12h), np.nan)
    
    # Calculate Camarilla for each 12h bar using lookback window
    for i in range(CAMARILLA_PERIOD, len(close_12h)):
        H = np.max(high_12h[i-CAMARILLA_PERIOD:i])
        L = np.min(low_12h[i-CAMARILLA_PERIOD:i])
        C = close_12h[i-1]  # previous close
        RANGE = H - L
        
        if RANGE > 0:  # avoid division by zero
            camarilla_R4[i] = C + (RANGE * 1.500)
            camarilla_R3[i] = C + (RANGE * 1.250)
            camarilla_S3[i] = C - (RANGE * 1.250)
            camarilla_S4[i] = C - (RANGE * 1.500)
        else:
            camarilla_R4[i] = camarilla_R3[i] = camarilla_S3[i] = camarilla_S4[i] = C
    
    # Align Camarilla levels to 6h timeframe
    camarilla_R4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_R4)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_S3)
    camarilla_S4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_S4)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(CAMARILLA_PERIOD, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_12h_slope_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(camarilla_R4_aligned[i]) or 
            np.isnan(camarilla_S4_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 12h EMA slope
        uptrend = ema_12h_slope_aligned[i] > 0
        downtrend = ema_12h_slope_aligned[i] < 0
        
        # Price levels
        r3 = camarilla_R3_aligned[i]
        s3 = camarilla_S3_aligned[i]
        r4 = camarilla_R4_aligned[i]
        s4 = camarilla_S4_aligned[i]
        
        # Trading logic based on regime
        # In ranging (low trend strength): fade at R3/S3
        # In trending: breakout at R4/S4 with volume
        
        # Ranging condition: weak trend (EMA slope near zero)
        ranging = np.abs(ema_12h_slope_aligned[i]) < 0.0001  # adjust based on price scale
        
        # Long signals
        long_signal = False
        if ranging and volume_ok:
            # Fade at S3 in ranging market
            long_signal = close[i] > s3 and close[i-1] <= s3  # bounce from S3
        elif not ranging and volume_ok:
            # Breakout at R4 in uptrend
            long_signal = uptrend and close[i] > r4 and close[i-1] <= r4
        
        # Short signals
        short_signal = False
        if ranging and volume_ok:
            # Fade at R3 in ranging market
            short_signal = close[i] < r3 and close[i-1] >= r3  # rejection at R3
        elif not ranging and volume_ok:
            # Breakdown at S4 in downtrend
            short_signal = downtrend and close[i] < s4 and close[i-1] >= s4
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit conditions
            exit_signal = False
            if ranging:
                # Exit long at R3 (fading target) or S4 breakdown
                if close[i] >= r3 or (not uptrend and close[i] < s4):
                    exit_signal = True
            else:
                # Exit long on trend reversal or S4 breakdown
                if not uptrend or close[i] < s4:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit conditions
            exit_signal = False
            if ranging:
                # Exit short at S3 (fading target) or R4 breakout
                if close[i] <= s3 or (not downtrend and close[i] > r4):
                    exit_signal = True
            else:
                # Exit short on trend reversal or R4 breakout
                if not downtrend or close[i] > r4:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals