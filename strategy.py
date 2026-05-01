#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray combo with 1w EMA50 trend filter and volume confirmation.
# Long when: Alligator bullish (jaw < teeth < lips) AND Elder Ray bull power > 0 AND price > 1w EMA50 (uptrend) AND volume > 1.5x 20-period volume median.
# Short when: Alligator bearish (jaw > teeth > lips) AND Elder Ray bear power < 0 AND price < 1w EMA50 (downtrend) AND volume > 1.5x 20-period volume median.
# Exit on opposite Alligator alignment (reversal) to reduce whipsaw.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-40 trades/year on 1d.
# Works in bull (buy Alligator uptrend with strength) and bear (sell Alligator downtrend with weakness).

name = "1d_WilliamsAlligator_ElderRay_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Williams Alligator (13,8,5 SMAs with future shifts)
    # Jaw: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward  
    # Lips: 5-period SMMA shifted 3 bars forward
    def smma(source, period):
        """Smoothed Moving Average"""
        if len(source) < period:
            return np.full_like(source, np.nan, dtype=np.float64)
        result = np.full_like(source, np.nan, dtype=np.float64)
        # First value is SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_PRICE) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)  # shift 8 bars forward
    teeth = np.roll(teeth, 5)  # shift 5 bars forward
    lips = np.roll(lips, 3)  # shift 3 bars forward
    
    # Calculate Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for Alligator and EMA
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: 1w EMA50 direction
        uptrend = curr_close > ema_50_1w_aligned[i]
        downtrend = curr_close < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        # Alligator conditions
        alligator_bullish = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])  # jaw < teeth < lips
        alligator_bearish = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])  # jaw > teeth > lips
        
        # Elder Ray conditions
        elder_bull = bull_power[i] > 0
        elder_bear = bear_power[i] < 0
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator bullish AND Elder Ray bull power > 0 AND uptrend AND volume confirmation
            if alligator_bullish and elder_bull and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND Elder Ray bear power < 0 AND downtrend AND volume confirmation
            elif alligator_bearish and elder_bear and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Alligator bearish (reversal signal)
            if alligator_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Alligator bullish (reversal signal)
            if alligator_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals