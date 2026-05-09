#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h Trend Filter + Volume Spike
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, with 12h EMA50 uptrend and volume spike
# Short when Bear Power < 0 and falling, Bull Power > 0 and rising, with 12h EMA50 downtrend and volume spike
# Works in bull/bear by adapting to trend via 12h EMA50 filter. Volume ensures momentum confirmation.
name = "6h_ElderRay_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Elder Ray components: EMA13 of close
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # 12h volume average for volume filter
    vol_12h = df_12h['volume'].values
    vol_avg_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 6h
    ema50_12h_6h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vol_avg_12h_6h = align_htf_to_ltf(prices, df_12h, vol_avg_12h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_12h_6h[i]) or np.isnan(vol_avg_12h_6h[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema50_12h_6h[i]
        vol_avg = vol_avg_12h_6h[i]
        vol_ok = volume[i] > vol_avg * 1.5
        
        # Elder Ray signals
        bull_rising = bull_power[i] > bull_power[i-1] if i > 0 else False
        bull_falling = bull_power[i] < bull_power[i-1] if i > 0 else False
        bear_rising = bear_power[i] > bear_power[i-1] if i > 0 else False
        bear_falling = bear_power[i] < bear_power[i-1] if i > 0 else False
        
        if position == 0:
            # Long: Bull Power > 0 and rising, Bear Power < 0, uptrend, volume spike
            if bull_power[i] > 0 and bull_rising and bear_power[i] < 0 and close[i] > trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and falling, Bull Power > 0, downtrend, volume spike
            elif bear_power[i] < 0 and bear_falling and bull_power[i] > 0 and close[i] < trend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power turns negative or trend reversal
            if bull_power[i] <= 0 or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power turns positive or trend reversal
            if bear_power[i] >= 0 or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals