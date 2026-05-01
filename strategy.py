#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + Elder Ray + Volume Spike
# Combines Alligator (Jaw/Teeth/Lips) for trend direction, Elder Ray (Bull/Bear Power) for momentum,
# and volume confirmation for institutional participation. Uses 1d HTF for Alligator smoothing.
# Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3) on median price.
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13).
# Long: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND volume > 1.5x 20-median.
# Short: Lips < Teeth < Jaw (bearish alignment) AND Bear Power < 0 AND volume > 1.5x 20-median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Designed for fewer trades (target 20-40/year) with high conviction entries in both bull and bear markets.

name = "4h_WilliamsAlligator_ElderRay_Volume_v1"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate 1d HTF data for Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Median price for Alligator: (high + low + close)/3
    median_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    mp_values = median_price.values
    
    # Alligator components: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs on median price
    jaw = pd.Series(mp_values).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(mp_values).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(mp_values).rolling(window=5, min_periods=5).mean().values
    
    # Shift Alligator components by 5 bars (future shift) to avoid look-ahead
    # Alligator is typically plotted with jaws/teeth/lips shifted into future
    jaw_shifted = np.concatenate([np.full(5, np.nan), jaw[:-5]])
    teeth_shifted = np.concatenate([np.full(5, np.nan), teeth[:-5]])
    lips_shifted = np.concatenate([np.full(5, np.nan), lips[:-5]])
    
    # Align shifted Alligator to LTF
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema_13 = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema_13
    bear_power = df_1d['low'].values - ema_13
    
    # Align Elder Ray to LTF (no extra delay needed for EMA-based)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for Alligator (max 13), ATR (14), volume (20)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(ema_13[i]) if i < len(ema_13) else True or  # placeholder, will be replaced by aligned check
            np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        # Alligator alignment: Lips > Teeth > Jaw (bullish) OR Lips < Teeth < Jaw (bearish)
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: bullish alignment AND Bull Power > 0 AND volume spike
            if bullish_alignment and bull_power_aligned[i] > 0 and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: bearish alignment AND Bear Power < 0 AND volume spike
            elif bearish_alignment and bear_power_aligned[i] < 0 and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Alligator turns bearish OR Bull Power <= 0
            elif not bullish_alignment or bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Alligator turns bullish OR Bear Power >= 0
            elif not bearish_alignment or bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals