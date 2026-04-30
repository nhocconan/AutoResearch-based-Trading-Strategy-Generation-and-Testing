#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla R1/S1 levels with volume confirmation and 4h trend filter
# Camarilla pivots identify key intraday support/resistance where institutional order flow clusters.
# Breakouts above R1 or below S1 with volume spike indicate strong institutional participation.
# 4h EMA(34) ensures alignment with intermediate-term trend to avoid counter-trend trades.
# Designed for low trade frequency (~30/year) to minimize fee drag in both bull and bear markets.
# Uses discrete position sizing (0.25) to reduce churn and ATR-based stops for risk control.

name = "4h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R1, S1, R2, S2)
    # Based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point (PP)
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Calculate Camarilla levels
    r1 = pp + (high_1d - low_1d) * 1.1 / 12.0
    s1 = pp - (high_1d - low_1d) * 1.1 / 12.0
    r2 = pp + (high_1d - low_1d) * 1.1 / 6.0
    s2 = pp - (high_1d - low_1d) * 1.1 / 6.0
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Calculate 4h EMA(34) for trend filter
    close_s = pd.Series(close)
    ema_34 = close_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # warmup for EMA(34)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.8x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (1.8 * vol_ma_20)
        
        curr_close = close[i]
        curr_ema = ema_34[i]
        curr_atr = atr[i]
        curr_r1 = r1_aligned[i]
        curr_s1 = s1_aligned[i]
        curr_r2 = r2_aligned[i]
        curr_s2 = s2_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 1d Camarilla R1 with 4h uptrend
                if curr_close > curr_r1 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 1d Camarilla S1 with 4h downtrend
                elif curr_close < curr_s1 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 1d Camarilla S2
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_s2:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1d Camarilla R2
            elif curr_close >= curr_r2:
                signals[i] = 0.0  # exit full position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 1d Camarilla R2
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_r2:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1d Camarilla S2
            elif curr_close <= curr_s2:
                signals[i] = 0.0  # exit full position
            else:
                signals[i] = -0.25
    
    return signals