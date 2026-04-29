#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d EMA34 trend filter and volume spike confirmation
# Elder Ray = Bull Power (High - EMA13) and Bear Power (Low - EMA13)
# Long when Bull Power > 0 AND Bear Power rising (less negative) in bullish 1d trend (price > EMA34)
# Short when Bear Power < 0 AND Bull Power falling (less positive) in bearish 1d trend (price < EMA34)
# Volume confirmation ensures institutional participation
# Target: 50-120 total trades over 4 years (12-30/year) to minimize fee drag
# Works in bull/bear via 1d trend filter - avoids counter-trend whipsaws

name = "6h_ElderRay_1dEMA34_Trend_VolumeSpike_v5"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Elder Ray components on 6h
    # EMA13 for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = Low - EMA13
    bear_power = low - ema_13
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20, 13)  # warmup for EMA34, volume MA, and EMA13
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema_34 = ema_34_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Determine 1d trend regime
        is_bull_trend = curr_close > curr_ema_34
        is_bear_trend = curr_close < curr_ema_34
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Long conditions: Bull Power positive AND Bear Power rising (less negative)
                # in bullish 1d trend
                if is_bull_trend and curr_bull_power > 0 and curr_bear_power > bear_power[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Short conditions: Bear Power negative AND Bull Power falling (less positive)
                # in bearish 1d trend
                elif is_bear_trend and curr_bear_power < 0 and curr_bull_power < bull_power[i-1]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when Bull Power turns negative OR Bear Power drops sharply
            if curr_bull_power <= 0 or curr_bear_power < bear_power[i-1] - 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when Bear Power turns positive OR Bull Power rises sharply
            if curr_bear_power >= 0 or curr_bull_power > bull_power[i-1] + 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals