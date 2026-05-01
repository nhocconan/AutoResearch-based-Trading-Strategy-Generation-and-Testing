#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h volume spike and 1d trend filter (EMA50 > EMA200).
# Uses 1d EMA crossover for bull/bear regime, 4h volume spike for momentum confirmation,
# and 1h Camarilla breakout for precise entry. Designed for low trade frequency (15-30/year)
# to minimize fee drag while capturing strong intraday moves in trending markets.
# Works in bull (long bias) and bear (short bias) via 1d regime filter.

name = "1h_Camarilla_R3S3_Breakout_4hVolumeSpike_1dEMA50_200_Trend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 1d EMAs for trend regime
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    # Bull regime: EMA50 > EMA200, Bear regime: EMA50 < EMA200
    bull_regime = ema50_1d > ema200_1d
    bear_regime = ema50_1d < ema200_1d
    
    # Align 1d regime to 1h timeframe
    bull_regime_aligned = align_htf_to_ltf(prices, df_1d, bull_regime)
    bear_regime_aligned = align_htf_to_ltf(prices, df_1d, bear_regime)
    
    # Load 4h data ONCE before loop for volume spike filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h volume average (20-period)
    vol_ma_4h = pd.Series(df_4h['volume']).rolling(window=20, min_periods=20).mean().values
    vol_spike_4h = df_4h['volume'].values > (vol_ma_4h * 2.0)  # 2x volume spike
    
    # Align 4h volume spike to 1h timeframe
    vol_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_spike_4h)
    
    # Calculate Camarilla levels (based on previous 1h bar's range)
    # We need the previous completed 1h bar for each 1h bar
    # Shift by 1 to avoid look-ahead: use prior bar's high/low/close
    camarilla_r3_1h = pd.Series(close).shift(1) + (pd.Series(high).shift(1) - pd.Series(low).shift(1)) * 1.1 / 4
    camarilla_s3_1h = pd.Series(close).shift(1) - (pd.Series(high).shift(1) - pd.Series(low).shift(1)) * 1.1 / 4
    
    # Volume confirmation: current 1h volume > 1.5x 20-bar average
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r3_1h[i]) or np.isnan(camarilla_s3_1h[i]) or np.isnan(vol_ma_1h[i]) or np.isnan(vol_spike_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_1h[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm_1h = curr_vol > (curr_vol_ma * 1.5)  # 1.5x volume spike
        
        # Camarilla breakout signals
        breakout_up = curr_high > camarilla_r3_1h[i]  # break above R3
        breakout_down = curr_low < camarilla_s3_1h[i]  # break below S3
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3 AND bull regime AND (1h vol spike OR 4h vol spike)
            if (breakout_up and 
                bull_regime_aligned[i] and 
                (volume_confirm_1h or vol_spike_4h_aligned[i])):
                signals[i] = 0.20
                position = 1
            # Short: breakout below S3 AND bear regime AND (1h vol spike OR 4h vol spike)
            elif (breakout_down and 
                  bear_regime_aligned[i] and 
                  (volume_confirm_1h or vol_spike_4h_aligned[i])):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S3 (stoploss) OR regime change to bear
            if (curr_low < camarilla_s3_1h[i] or 
                bear_regime_aligned[i]):  # exit if bear regime starts
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (stoploss) OR regime change to bull
            if (curr_high > camarilla_r3_1h[i] or 
                bull_regime_aligned[i]):  # exit if bull regime starts
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals