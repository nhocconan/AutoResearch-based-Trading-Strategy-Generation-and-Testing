#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation
# Uses 4h/1d for signal direction (trend/regime), 1h only for entry timing precision.
# Camarilla levels provide institutional support/resistance; breakouts with volume confirm momentum.
# Works in bull/bear markets by following 4h trend while capturing 1h breakouts.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
# Session filter (08-20 UTC) reduces noise trades.

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) for filtering
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop for 4h and 1d calculations
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d EMA(34) for stronger trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for 1h timeframe using previous day's OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.25*(high-low), 
    #            R2 = close + 1.1*(high-low), R1 = close + 0.5*(high-low)
    #            S1 = close - 0.5*(high-low), S2 = close - 1.1*(high-low), 
    #            S3 = close - 1.25*(high-low), S4 = close - 1.5*(high-low)
    # We use daily OHLC from 1d timeframe to calculate Camarilla for 1h
    # For each 1h bar, we use the previous completed 1d bar's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_for_camarilla = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    camarilla_calc = high_1d - low_1d
    R1 = close_1d_for_camarilla + 0.5 * camarilla_calc
    S1 = close_1d_for_camarilla - 0.5 * camarilla_calc
    # Align to 1h timeframe (using previous completed 1d bar)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34, 20)  # warmup for EMAs and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available or outside session
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]):
            signals[i] = 0.0
            continue
            
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_R1 = R1_aligned[i]
        curr_S1 = S1_aligned[i]
        curr_ema50_4h = ema50_4h_aligned[i]
        curr_ema34_1d = ema34_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > both 4h EMA50 AND 1d EMA34
        is_bullish_regime = curr_close > curr_ema50_4h and curr_close > curr_ema34_1d
        # Bearish if price < both 4h EMA50 AND 1d EMA34
        is_bearish_regime = curr_close < curr_ema50_4h and curr_close < curr_ema34_1d
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: price breaks above R1 AND bullish regime
                if curr_close > curr_R1 and is_bullish_regime:
                    signals[i] = 0.20
                    position = 1
                # Bearish entry: price breaks below S1 AND bearish regime
                elif curr_close < curr_S1 and is_bearish_regime:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price breaks below S1 OR regime changes to bearish
            if curr_close < curr_S1 or not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price breaks above R1 OR regime changes to bullish
            if curr_close > curr_R1 or not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals