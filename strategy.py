# [EXPERIMENT #69031] - 6h_Camarilla_Pivot_R1S1_Breakout_VolumeFilter_v1
# Hypothesis: Camarilla pivot levels from 1d provide robust support/resistance zones.
# Breakouts above R1 (bullish) or below S1 (bearish) with volume confirmation capture
# institutional flow. Works in bull/bear by filtering with 1d EMA50 trend and
# avoiding chop with BB width ratio > 0.8. Targets 15-35 trades/year.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE for Camarilla pivots, trend, volatility, and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ATR(14) for volatility regime
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1d Volume ratio (current / 20-period average) for confirmation
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / np.where(vol_ma_20_1d == 0, 1, vol_ma_20_1d)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # 1d Bollinger Bands width ratio for regime detection
    bb_period = 20
    bb_std = 2.0
    bb_middle = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_middle + bb_std * bb_std_dev
    bb_lower = bb_middle - bb_std * bb_std_dev
    bb_width = bb_upper - bb_lower
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_width_ratio = bb_width / np.where(bb_width_ma == 0, 1, bb_width_ma)
    bb_width_ratio_aligned = align_htf_to_ltf(prices, df_1d, bb_width_ratio)
    
    # Camarilla pivot levels from previous 1d OHLC
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), R2 = C + ((H-L) * 1.1/6), R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12), S2 = C - ((H-L) * 1.1/6), S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    camarilla_HL = high_1d - low_1d
    camarilla_R1 = close_1d + camarilla_HL * 1.1 / 12
    camarilla_S1 = close_1d - camarilla_HL * 1.1 / 12
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # 6h price data
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(bb_width_ratio_aligned[i]) or
            np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend = ema_50_1d_aligned[i]
        atr = atr_14_1d_aligned[i]
        vol_ratio = vol_ratio_1d_aligned[i]
        bb_width_ratio = bb_width_ratio_aligned[i]
        r1 = camarilla_R1_aligned[i]
        s1 = camarilla_S1_aligned[i]
        
        # Trend filter: price relative to daily EMA
        trend_up = price > ema_trend
        trend_down = price < ema_trend
        
        # Regime filter: avoid extreme chop (BB width too narrow)
        regime_filter = bb_width_ratio > 0.8  # Not in extreme squeeze
        
        # Volatility filter: avoid extreme volatility spikes
        atr_ma_20 = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = (atr > 0.3 * atr_ma_20) and (atr < 3.0 * atr_ma_20)
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio > 1.3)
        
        if position == 0:
            # Enter long when price breaks above R1 with trend, volume, and regime
            if price > r1 and trend_up and vol_filter and regime_filter:
                signals[i] = 0.25
                position = 1
            # Enter short when price breaks below S1 with trend, volume, and regime
            elif price < s1 and trend_down and vol_filter and regime_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal, volatility spike, or price returns below R1
            if not trend_up or price < r1 or (atr > 3.5 * atr_ma_20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reversal, volatility spike, or price returns above S1
            if not trend_down or price > s1 or (atr > 3.5 * atr_ma_20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_Pivot_R1S1_Breakout_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0