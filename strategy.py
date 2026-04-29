#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Camarilla pivot levels provide strong intraday support/resistance. Breakout of R3/S3
# indicates strong momentum. Combined with 1d EMA34 trend filter ensures we trade
# with the higher timeframe trend. Volume spike confirms participation.
# Uses 4h timeframe targeting 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
# Works in bull/bear markets by following 1d trend while capturing 4h momentum breakouts.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe (completed 1d bar only)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels for 4h (using typical price)
    typical_price = (high + low + close) / 3.0
    # Use previous bar's data for pivot calculation (no look-ahead)
    prev_typical = pd.Series(typical_price).shift(1)
    prev_high = pd.Series(high).shift(1)
    prev_low = pd.Series(low).shift(1)
    prev_close = pd.Series(close).shift(1)
    
    # Camarilla calculations: based on previous bar's range
    cam_pp = (prev_high + prev_low + prev_close) / 3.0
    cam_range = prev_high - prev_low
    # Resistance levels
    cam_r3 = cam_pp + (cam_range * 1.1 / 2)
    cam_r4 = cam_pp + (cam_range * 1.1)
    # Support levels
    cam_s3 = cam_pp - (cam_range * 1.1 / 2)
    cam_s4 = cam_pp - (cam_range * 1.1)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20) + 1  # warmup for EMA34, volume MA, and shift
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema34_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Get Camarilla levels from previous bar (already calculated)
        r3 = cam_r3[i]
        s3 = cam_s3[i]
        
        # Trend regime: bullish if price > 1d EMA34, bearish if price < 1d EMA34
        is_bullish_regime = curr_close > ema34_aligned[i]
        is_bearish_regime = curr_close < ema34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: break above R3 AND bullish regime
                if curr_high > r3 and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: break below S3 AND bearish regime
                elif curr_low < s3 and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price drops below S3 OR regime changes to bearish
            if curr_low < s3 or not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price rises above R3 OR regime changes to bullish
            if curr_high > r3 or not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals