#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray + TRIX confluence with 1w trend filter.
# Long when: price > Alligator Jaw, Bull Power > 0, TRIX > 0 AND 1w EMA50 uptrend.
# Short when: price < Alligator Jaw, Bear Power < 0, TRIX < 0 AND 1w EMA50 downtrend.
# Uses 1d timeframe for lower trade frequency (~15-25 trades/year) and 1w for major trend alignment.
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend direction and alignment.
# Elder Ray measures bull/bear power behind price moves. TRIX filters noise and confirms momentum.
# Volume confirmation added to ensure breakout conviction. Target: 30-100 total trades over 4 years.

name = "1d_Alligator_ElderRay_TRIX_1wEMA50_Volume_v1"
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) - all SMMA
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # TRIX: EMA(EMA(EMA(close,15),15),15) - 1 period ROC
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = np.full_like(close, np.nan, dtype=np.float64)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    
    # Volume confirmation: 20-period volume median
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(trix[i]) or np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1w EMA50 direction
        uptrend = curr_close > ema_50_1w_aligned[i]
        downtrend = curr_close < ema_50_1w_aligned[i]
        
        # Alligator alignment: Jaw > Teeth > Lips (uptrend) or Jaw < Teeth < Lips (downtrend)
        alligator_long = jaw[i] > teeth[i] > lips[i]
        alligator_short = jaw[i] < teeth[i] < lips[i]
        
        # Elder Ray confirmation
        bull_confirm = bull_power[i] > 0
        bear_confirm = bear_power[i] < 0
        
        # TRIX confirmation
        trix_long = trix[i] > 0
        trix_short = trix[i] < 0
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator uptrend AND Bull Power > 0 AND TRIX > 0 AND 1w uptrend AND volume spike
            if alligator_long and bull_confirm and trix_long and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend AND Bear Power < 0 AND TRIX < 0 AND 1w downtrend AND volume spike
            elif alligator_short and bear_confirm and trix_short and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns down OR Bull Power <= 0 OR TRIX <= 0 OR 1w trend turns down
            if (not alligator_long) or (bull_power[i] <= 0) or (trix[i] <= 0) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns up OR Bear Power >= 0 OR TRIX >= 0 OR 1w trend turns up
            if (not alligator_short) or (bear_power[i] >= 0) or (trix[i] >= 0) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals