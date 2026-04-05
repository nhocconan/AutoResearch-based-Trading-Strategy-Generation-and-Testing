#!/usr/bin/env python3
"""
Experiment #8171: 6-hour ADX + Williams Alligator combination with 1-day trend filter.
Hypothesis: In trending markets (ADX > 25), price trading outside the Williams Alligator 
(teeth/lips) with alignment to 1-day trend captures sustained moves. The Alligator acts as 
a dynamic trend filter - when jaws, teeth, lips are aligned and price is outside their 
range, it indicates strong momentum. Works in both bull/bear by using ADX for trend strength 
and Alligator for direction, avoiding whipsaws in ranging markets (ADX < 20).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8171_6h_adx_alligator_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
ADX_THRESHOLD = 25
ALLIGATOR_PERIOD_JAW = 13  # smoothed SMA
ALLIGATOR_PERIOD_TEETH = 8
ALLIGATOR_PERIOD_LIPS = 5
JAW_OFFSET = 8
TEETH_OFFSET = 5
LIPS_OFFSET = 3
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
TREND_PERIOD = 50

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=TREND_PERIOD, adjust=False, min_periods=TREND_PERIOD).mean().values
    price_vs_ema = np.where(close_1d > ema_1d, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator: SMMA (smoothed moving average)
    # Jaw (Blue): 13-period SMMA, 8 periods ahead
    # Teeth (Red): 8-period SMMA, 5 periods ahead  
    # Lips (Green): 5-period SMMA, 3 periods ahead
    def smma(series, period):
        """Smoothed Moving Average"""
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean()
        # First value is SMA, then smoothed
        result = np.full_like(series, np.nan, dtype=float)
        if len(sma) >= period:
            result[period-1] = sma.iloc[period-1]
            for i in range(period, len(series)):
                if not np.isnan(sma.iloc[i]) and not np.isnan(result[i-1]):
                    result[i] = (sma.iloc[i] + result[i-1] * (period-1)) / period
                else:
                    result[i] = result[i-1] if not np.isnan(result[i-1]) else np.nan
        return result
    
    jaw = smma(high, ALLIGATOR_PERIOD_JAW)  # Using high for jaw
    teeth = smma(low, ALLIGATOR_PERIOD_TEETH)  # Using low for teeth
    lips = smma(close, ALLIGATOR_PERIOD_LIPS)  # Using close for lips
    
    # Apply offsets (shift forward)
    jaw = np.roll(jaw, -JAW_OFFSET)
    teeth = np.roll(teeth, -TEETH_OFFSET)
    lips = np.roll(lips, -LIPS_OFFSET)
    # Set NaN for rolled values
    jaw[-JAW_OFFSET:] = np.nan
    teeth[-TEETH_OFFSET:] = np.nan
    lips[-LIPS_OFFSET:] = np.nan
    
    # ADX calculation
    def calculate_adx(high, low, close, period):
        """Calculate ADX (Average Directional Index)"""
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
            minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
            
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Wilder's smoothing
        atr = np.zeros(len(high))
        atr[0] = tr[0]
        for i in range(1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros(len(high))
        minus_di = np.zeros(len(high))
        dx = np.zeros(len(high))
        
        for i in range(period, len(high)):
            if atr[i] != 0:
                plus_di[i] = (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().iloc[i] / atr[i]) * 100
                minus_di[i] = (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().iloc[i] / atr[i]) * 100
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100
        
        adx = np.zeros(len(high))
        adx[2*period-1] = np.mean(dx[period:2*period]) if np.sum(~np.isnan(dx[period:2*period])) >= period else np.nan
        for i in range(2*period, len(high)):
            if not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            else:
                adx[i] = adx[i-1]
        
        return adx
    
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
    start = max(
        ALLIGATOR_PERIOD_JAW + JAW_OFFSET,
        ALLIGATOR_PERIOD_TEETH + TEETH_OFFSET,
        ALLIGATOR_PERIOD_LIPS + LIPS_OFFSET,
        ADX_PERIOD * 2,
        ATR_PERIOD,
        TREND_PERIOD
    ) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1 and i < len(close) and close[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and i < len(close) and close[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if Alligator lines not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Alligator alignment: jaws > teeth > lips (bullish) or jaws < teeth < lips (bearish)
        bullish_alignment = jaw[i] > teeth[i] > lips[i]
        bearish_alignment = jaw[i] < teeth[i] < lips[i]
        
        # Price outside Alligator mouth
        price_above_alligator = close[i] > jaw[i] and close[i] > teeth[i]
        price_below_alligator = close[i] < jaw[i] and close[i] < teeth[i]
        
        # ADX trend strength
        strong_trend = adx[i] > ADX_THRESHOLD
        
        # Entry conditions
        long_entry = bullish_alignment and price_above_alligator and strong_trend and price_vs_ema_aligned[i] == 1
        short_entry = bearish_alignment and price_below_alligator and strong_trend and price_vs_ema_aligned[i] == -1
        
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