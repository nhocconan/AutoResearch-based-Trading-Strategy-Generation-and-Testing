# 1d_1wPivot_PriceAction_VolumeBreak
# Hypothesis: 1-day timeframe with weekly pivot points from weekly data (HTF)
# - Weekly pivot points (from actual weekly candles) provide institutional support/resistance
# - Price breaking above weekly S1 with volume and daily uptrend = long
# - Price breaking below weekly R1 with volume and daily downtrend = short
# - Uses daily timeframe to reduce trade frequency and avoid fee drag
# - Volume confirmation (2x average) filters false breakouts
# - Works in bull (buy dips at S1 in uptrend) and bear (sell rallies at R1 in downtrend)
# - Exit when price returns to weekly pivot (PP) or volume weakens
# - Target: 10-25 trades/year to stay under fee drag threshold

name = "1d_1wPivot_PriceAction_VolumeBreak"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need at least 2-3 weeks for pivot calculation
        return np.zeros(n)
    
    # Weekly pivot points from actual weekly candles
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Classic pivot point calculation
    pp = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to daily timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume spike detection: 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 2.0
            uptrend = ema_34[i] > ema_34[i-1]
            
            if close[i] > s1_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below R1 with volume and daily downtrend
            elif close[i] < r1_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below pivot or volume drops
            if close[i] < pp_aligned[i] or volume[i] < vol_ma_20[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above pivot or volume drops
            if close[i] > pp_aligned[i] or volume[i] < vol_ma_20[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Note: Uses actual weekly pivot points from weekly candles (not approximated from daily)
# Weekly pivot provides structure that works across market regimes
# Position size 0.25 targets 10-25 trades/year, avoiding fee drag
# Volume confirmation (2x average) filters false breakouts
# Daily EMA(34) trend filter ensures alignment with intermediate trend
# Exit at weekly pivot (PP) captures mean reversion to institutional levels
# Works in bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)