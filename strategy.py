#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with Daily EMA34 Trend and Volume Spike
# Uses Elder Ray indicators (Bull Power = High - EMA13, Bear Power = Low - EMA13) to measure
# buying/selling pressure relative to short-term trend. Combined with daily EMA34 for
# higher timeframe trend filter and volume confirmation to avoid false signals.
# Works in bull markets via strong Bull Power + uptrend, and in bear markets via
# strong Bear Power + downtrend. Volume spike ensures momentum confirmation.
# Target: 12-25 trades/year (50-100 total over 4 years)

name = "6h_ElderRay_1dEMA34_Trend_VolumeSpike_v1"
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
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Elder Ray on 6h timeframe: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Align daily EMA34 to 6h timeframe (completed 1d bar only)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20, 13)  # warmup for EMA34, volume MA, and EMA13
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema34 = ema_34_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Determine daily trend: price above/below EMA34
        uptrend = curr_close > curr_ema34
        downtrend = curr_close < curr_ema34
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and in direction of daily trend
            if curr_volume_confirm:
                # Bullish entry: strong bull power + uptrend
                if curr_bull_power > 0 and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: strong bear power + downtrend
                elif curr_bear_power < 0 and downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit when bull power fades or trend changes
            # Exit when bull power turns negative OR price breaks below daily EMA34
            if curr_bull_power <= 0 or curr_close < curr_ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit when bear power fades or trend changes
            # Exit when bear power turns positive OR price breaks above daily EMA34
            if curr_bear_power >= 0 or curr_close > curr_ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals