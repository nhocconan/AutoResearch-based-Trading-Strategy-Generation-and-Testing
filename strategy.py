#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d EMA50 trend filter + volume spike confirmation
# Williams Alligator identifies trend absence/presence via three smoothed MAs (Jaw, Teeth, Lips).
# In ranging markets (Alligator sleeping), we fade extremes; in trending markets (Alligator awakening), we follow the trend.
# Combined with 1d EMA50 for higher-timeframe trend bias and volume spike for confirmation.
# Target: 20-50 trades/year (80-200 over 4 years) to minimize fee drag and avoid overtrading.

name = "4h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator components: SMMA (Smoothed Moving Average) = EMA with alpha=1/period
    # Jaw: SMMA(13, 8) -> EMA(13) then smoothed by 8 periods
    # Teeth: SMMA(8, 5) -> EMA(8) then smoothed by 5 periods
    # Lips: SMMA(5, 3) -> EMA(5) then smoothed by 3 periods
    # We approximate SMMA with EMA for simplicity and computational efficiency
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Alligator conditions
    # Alligator sleeping (ranging): jaws, teeth, lips intertwined
    # Alligator awakening (trending): lips > teeth > jaw (uptrend) or lips < teeth < jaw (downtrend)
    lips_above_teeth = lips > teeth
    teeth_above_jaw = teeth > jaw
    lips_below_teeth = lips < teeth
    teeth_below_jaw = teeth < jaw
    
    # Uptrend: lips > teeth > jaw
    uptrend_alligator = lips_above_teeth & teeth_above_jaw
    # Downtrend: lips < teeth < jaw
    downtrend_alligator = lips_below_teeth & teeth_below_jaw
    # Ranging: otherwise (Alligator sleeping)
    ranging = ~(uptrend_alligator | downtrend_alligator)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20, 13, 8, 5)  # Need sufficient history for 1d EMA, volume MA, and Alligator
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Alligator conditions
        uptrend_alli = uptrend_alligator[i]
        downtrend_alli = downtrend_alligator[i]
        ranging_i = ranging[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # In ranging market: fade extremes (mean reversion)
            # Long when price touches lower Bollinger Band proxy (low < lips - 0.5*ATR)
            # Short when price touches upper Bollinger Band proxy (high > lips + 0.5*ATR)
            # We approximate ATR with high-low range for simplicity
            atr_proxy = (high[i] - low[i])  # Simple range proxy
            
            # Fade long: price near lower extreme in ranging market
            fade_long = ranging_i and (low[i] < lips[i] - 0.5 * atr_proxy) and vol_spike
            # Fade short: price near upper extreme in ranging market
            fade_short = ranging_i and (high[i] > lips[i] + 0.5 * atr_proxy) and vol_spike
            
            # In trending market: follow the trend
            # Long when uptrend Alligator aligned with 1d EMA50 trend
            trend_long = uptrend_alli and (close[i] > ema_50_1d_aligned[i]) and vol_spike
            # Short when downtrend Alligator aligned with 1d EMA50 trend
            trend_short = downtrend_alli and (close[i] < ema_50_1d_aligned[i]) and vol_spike
            
            if fade_long or trend_long:
                signals[i] = 0.25
                position = 1
            elif fade_short or trend_short:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions:
            # 1. Alligator signals ranging (sleeping) - take profit
            # 2. Trend reversal: price closes below 1d EMA50
            # 3. Alligator signals downtrend
            if ranging_i or (close[i] < ema_50_1d_aligned[i]) or downtrend_alli:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Alligator signals ranging (sleeping) - take profit
            # 2. Trend reversal: price closes above 1d EMA50
            # 3. Alligator signals uptrend
            if ranging_i or (close[i] > ema_50_1d_aligned[i]) or uptrend_alli:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals