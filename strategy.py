#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d with breakout/fade logic
# - Fade at R3/S3 (mean reversion in range) during low volatility (choppy markets)
# - Breakout continuation at R4/S4 (trend following) during high volatility (trending markets)
# - Volume confirmation to avoid false breakouts
# - Designed to work in both bull (breakouts) and bear (fades) markets
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Using formula: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # where C = (H+L+CLOSE)/3 of previous day
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Pivot point
    p = (prev_high + prev_low + prev_close) / 3
    # Range
    r = prev_high - prev_low
    
    # Camarilla levels
    r4 = p + (r * 1.1 / 2)
    r3 = p + (r * 1.1 / 4)
    s3 = p - (r * 1.1 / 4)
    s4 = p - (r * 1.1 / 2)
    
    # Align to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Get weekly data for trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volatility filter using ATR(14) from weekly
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Volume confirmation: volume > 1.5x average
    volume_1d = df_1d['volume'].values
    vol_avg_10 = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    vol_avg_10_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_10)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr_14_1w_aligned[i]) or
            np.isnan(vol_avg_10_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Volatility regime: high vol = trending, low vol = ranging
        atr_median = np.median(atr_14_1w_aligned[max(0, i-49):i+1]) if i >= 50 else atr_14_1w_aligned[i]
        high_vol = atr_14_1w_aligned[i] > atr_median * 1.2
        low_vol = atr_14_1w_aligned[i] <= atr_median * 1.2
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_avg_10_aligned[i]
        
        # Trend filter
        price_above_weekly_ema = close[i] > ema_20_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_20_1w_aligned[i]
        
        # Fade logic (low vol): sell at R3, buy at S3
        if low_vol:
            # Fade at R3/S3
            fade_long = (close[i] <= s3_aligned[i]) and vol_confirm
            fade_short = (close[i] >= r3_aligned[i]) and vol_confirm
            
            if fade_long and position <= 0:
                signals[i] = 0.25
                position = 1
            elif fade_short and position >= 0:
                signals[i] = -0.25
                position = -1
            # Exit fade on reversion to mean (pivot)
            elif position == 1 and close[i] >= p[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] <= p[i]:
                signals[i] = 0.0
                position = 0
            # Hold fade position
            else:
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        
        # Breakout logic (high vol): break R4/S4
        else:
            # Breakout continuation
            breakout_long = (close[i] > r4_aligned[i]) and price_above_weekly_ema and vol_confirm
            breakout_short = (close[i] < s4_aligned[i]) and price_below_weekly_ema and vol_confirm
            
            if breakout_long and position <= 0:
                signals[i] = 0.25
                position = 1
            elif breakout_short and position >= 0:
                signals[i] = -0.25
                position = -1
            # Exit breakout on trend reversal (weekly EMA cross)
            elif position == 1 and not price_above_weekly_ema:
                signals[i] = 0.0
                position = 0
            elif position == -1 and not price_below_weekly_ema:
                signals[i] = 0.0
                position = 0
            # Hold breakout position
            else:
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R3S3_R4S4_VolVol"
timeframe = "6h"
leverage = 1.0