#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout + 12h EMA34 Trend + Volume Spike + Chop Filter
# Camarilla R3/S3 are stronger support/resistance levels than R1/S1, reducing false breakouts.
# Requires alignment with 12h EMA34 trend, volume > 1.8x 20-bar average, and choppy market filter (CHOP > 50).
# Chop filter avoids trending markets where Camarilla levels are less effective for mean reversion.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 75-200 total trades over 4 years (19-50/year).
# Works in both bull/bear markets by combining mean-reversion at strong levels with trend filter.

name = "4h_Camarilla_R3_S3_Breakout_12hEMA34_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA(34) for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Camarilla levels from prior 12h bar (represents prior day for 4h chart)
    prior_high = df_12h['high'].shift(1).values
    prior_low = df_12h['low'].shift(1).values
    prior_close = df_12h['close'].shift(1).values
    
    # Camarilla: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    pp = (prior_high + prior_low + prior_close) / 3.0
    r3 = prior_close + (prior_high - prior_low) * 1.1 / 4.0
    s3 = prior_close - (prior_high - prior_low) * 1.1 / 4.0
    
    # Align Camarilla levels to 4h
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    # Choppiness Index filter: CHOP > 50 indicates choppy/range-bound market (good for mean reversion)
    # CHOP = 100 * log10(sum(ATR(14) over 14 periods) / log10(highest high - lowest low over 14 periods))
    atr_period = 14
    chop_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # First bar: no previous close
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum().values
    hh = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    ll = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100 * np.log10(atr * chop_period / np.log10(range_hl)) / np.log10(chop_period)
    chop_filter = chop > 50.0  # Choppy market favors mean reversion
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, chop_period)  # Ensure sufficient history
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(volume_ma_20[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_confirm[i]
        chop_ok = chop_filter[i]
        ema_trend_up = close[i] > ema_34_12h_aligned[i]
        ema_trend_down = close[i] < ema_34_12h_aligned[i]
        
        price = close[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price > R3, 12h EMA34 uptrend, volume confirm, choppy market
            if price > r3_aligned[i] and ema_trend_up and vol_confirm and chop_ok:
                signals[i] = 0.25
                position = 1
            # Short entry: Price < S3, 12h EMA34 downtrend, volume confirm, choppy market
            elif price < s3_aligned[i] and ema_trend_down and vol_confirm and chop_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on retracement to PP or below S3
            if price < pp_aligned[i] or price < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on retracement to PP or above R3
            if price > pp_aligned[i] or price > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals