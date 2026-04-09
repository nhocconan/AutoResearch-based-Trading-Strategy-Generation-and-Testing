#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot levels with volume-weighted RSI and ATR-based volatility filter
# Weekly Camarilla levels (R3/S3, R4/S4) act as major support/resistance in both bull and bear markets
# Volume-weighted RSI (VW-RSI) filters false signals by incorporating volume into momentum
# ATR filter ensures sufficient volatility (avoid choppy low-vol periods)
# Fade at weekly R3/S3 (mean reversion), breakout at weekly R4/S4 (trend following)
# Position size fixed at 0.25 to balance return and drawdown
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1w_vwrs_camarilla_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly Camarilla pivot levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r4_1w = close_1w + range_1w * 1.1 / 2.0
    r3_1w = close_1w + range_1w * 1.1 / 4.0
    s3_1w = close_1w - range_1w * 1.1 / 4.0
    s4_1w = close_1w - range_1w * 1.1 / 2.0
    
    # Calculate 1w ATR (14-period) for volatility filtering
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 6h timeframe
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Pre-compute volume-weighted RSI (14-period) for 6h
    # VW-RSI = RSI where gains/losses are weighted by volume
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    vol_weighted_gain = gain * volume
    vol_weighted_loss = loss * volume
    avg_vw_gain = pd.Series(vol_weighted_gain).rolling(window=14, min_periods=14).mean().values
    avg_vw_loss = pd.Series(vol_weighted_loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_vw_loss != 0, avg_vw_gain / avg_vw_loss, 0)
    vw_rsi = 100 - (100 / (1 + rs))
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(atr_1w_aligned[i]) or np.isnan(vw_rsi[i]) or
            np.isnan(vol_ma_20[i]) or not in_session[i] or
            atr_1w_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.3x average 6h volume
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # Volatility filter: only trade when 1w ATR is above its 50-period average
        atr_ma_50_1w = pd.Series(atr_1w_aligned).rolling(window=50, min_periods=50).mean()
        if len(atr_ma_50_1w) > i:
            vol_filter = atr_1w_aligned[i] > atr_ma_50_1w.iloc[i]
        else:
            vol_filter = True  # Not enough data for MA, allow trading
            
        if not vol_filter:
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on retracement to S3 (1w) or stop at S4 breakdown (1w)
            if close[i] < s3_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] < s4_1w_aligned[i]:  # Stop loss at S4 breakdown
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to R3 (1w) or stop at R4 breakout (1w)
            if close[i] > r3_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] > r4_1w_aligned[i]:  # Stop loss at R4 breakout
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Camarilla pivot trading with volume and volatility confirmation
            # Fade at R3/S3 (mean reversion), breakout at R4/S4 (trend following)
            if volume_confirmed:
                # Fade at R3 (sell at resistance, expect reversion to pivot)
                if close[i] > r3_1w_aligned[i] and close[i] < r4_1w_aligned[i]:
                    position = -1
                    signals[i] = -position_size
                # Fade at S3 (buy at support, expect reversion to pivot)
                elif close[i] < s3_1w_aligned[i] and close[i] > s4_1w_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Breakout continuation at R4 (buy break above resistance)
                elif close[i] > r4_1w_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Breakout continuation at S4 (sell break below support)
                elif close[i] < s4_1w_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals